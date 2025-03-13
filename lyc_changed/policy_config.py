import logging
from typing import Any, TypeAlias

import dataclasses
import pathlib

import sentencepiece
import numpy as np

from openpi.training import config as _config
import openpi.transforms as _transforms
from openpi.shared import array_typing as at


# center crop
import einops
import json

import lyc_changed.policy as _policy

from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor
from transformers.image_transforms import resize, center_crop

import numpydantic
import pydantic

@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile
    mask: numpydantic.NDArray | None = None

DataDict: TypeAlias = at.PyTree


@dataclasses.dataclass(frozen=True)
class Normalize(_transforms.DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    # def __post_init__(self):
    #     if self.norm_stats is not None and self.use_quantiles:
    #         _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return _transforms.apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(_transforms.DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    # def __post_init__(self):
    #     if self.norm_stats is not None and self.use_quantiles:
    #         _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return _transforms.apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        if stats.mask is None:
            mask = np.ones_like(stats.mean, dtype=bool)
        else:
            mask = stats.mask
        actions = np.where(
                mask,
                (x * (stats.std[np.newaxis,:] + 1e-8) + stats.mean[np.newaxis,:]),
                0.5 * (x + 1), # specialize for gripper
            )
        return actions

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        if stats.mask is None:
            mask = np.ones_like(stats.mean, dtype=bool)
        else:
            mask = stats.mask   
        return np.where(
                mask,
                (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01,
                0.5 * (x + 1), # specialize for gripper
            )


@dataclasses.dataclass(frozen=True)
class ProcessImage(_transforms.DataTransformFn):
    processor: PaliGemmaProcessor
    max_len: int = 50
    def __call__(self,  data) -> np.ndarray:
        # processor = PaliGemmaProcessor.from_pretrained(HF_HUB_REPO, token=hf_token, local_files_only=True)
        image = {}
        for k, v in data['image'].items():
            v = center_crop(image=v, size=(224*(0.9**0.5), 224*(0.9**0.5)))
            v = resize(image=v, size=(224, 224))
            v = self.processor.image_processor(images=v, return_tensors="np")["pixel_values"][0]
            if v.shape[0] == 3:
                v = einops.rearrange(v, "c h w -> h w c")            
            image[k] = v
            
        data['image'] = image
        
        # tokenize prompt
        tokens, token_masks = self.tokenize(data['prompt'])
        data.pop('prompt')
        data = {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
        
        return data
    
    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.processor.tokenizer(prompt, truncation=True, return_tensors="np", add_special_tokens=True).input_ids[0]
        tokens = list(tokens)
        tokens_len = len(tokens)
        if tokens_len < self.max_len:
            padding = [False] * (self.max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self.max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self.max_len]
            mask = [True] * self.max_len

        return np.asarray(tokens), np.asarray(mask)


def load_norm_stats(checkpoint_dir: pathlib.Path | str, unnorm_key: str = "unnorm") -> dict[str, _transforms.NormStats]:
    path = pathlib.Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    norm_dict = json.loads(path.read_text())[unnorm_key]
    new_norm_dict = {}
    for k, v in norm_dict.items():
        if k == "proprio":
            new_norm_dict["state"] = NormStats(**v)
        elif k == "action":
            new_norm_dict["actions"] = NormStats(**v)
    return new_norm_dict


def create_torch_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    unnorm_key: str | None = 'libero_spatial_no_noops',
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or _transforms.Group()
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    logging.info("Loading model...")
    # if (checkpoint_dir / "params").exists():
    #     model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    # else:
    #     # convert torch checkpoint to jax checkpoint
    import torch
    params = torch.load(checkpoint_dir / "jax_params.pt", weights_only=False)
    model = train_config.model.load(params)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    assert norm_stats is not None
    # if norm_stats is None:
    # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
    # that the policy is using the same normalization stats as the original training process.
    norm_stats = load_norm_stats(norm_stats, unnorm_key)
    
    
    # import ipdb; ipdb.set_trace()

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
