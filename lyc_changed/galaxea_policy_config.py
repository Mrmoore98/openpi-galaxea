import dataclasses
import pathlib

import openpi.transforms as _transforms
from openpi.models import model as _model
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import DataConfigFactory, DataConfig, GroupFactory, TrainConfig
from typing_extensions import override

from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor
# from lyc_changed.galaxea_policy import GalaxeaInputs, GalaxeaOutputs
from lyc_changed.libero_policy import LiberoInputs, LiberoOutputs
import lyc_changed.pi0 as pi0
from lyc_changed.policy_config import ProcessImage



@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    processor: PaliGemmaProcessor = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
                inputs=[
                    # _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    ProcessImage(processor=self.processor, max_len=model_config.max_token_len),
                ],
            )



@dataclasses.dataclass(frozen=True)
class GalaxeaDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        # data_transforms = _transforms.Group(
        #     inputs=[GalaxeaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
        #     outputs=[GalaxeaOutputs()],
        # )
        data_transforms = _transforms.Group(
            inputs=[LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[LiberoOutputs()],
        )
        # Use delta actions (not for gripper)
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(
            processor=PaliGemmaProcessor.from_pretrained(
                'google/paligemma2-3b-pt-224', 
                token='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', 
                local_files_only=True
            )
        )(model_config)

        return dataclasses.replace(
            self.base_config or DataConfig(),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


galaxea_zero_config = TrainConfig(
    name="pi0_libero",
    model=pi0.Pi0Config(action_dim=7),
    data=GalaxeaDataConfig(
        repo_id="galaxea/libero",
        base_config=DataConfig(
            local_files_only=True, #[yc] changed  # Set to True for local-only datasets.
            prompt_from_task=True,
            use_quantile_norm=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
