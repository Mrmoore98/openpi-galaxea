import dataclasses
import enum
import logging
import socket

import tyro
import pathlib

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server

import sys
sys.path.append("/EFM-Pretrain/lyc/openpi")
from lyc_changed.policy_config import create_torch_policy
from lyc_changed.galaxea_policy_config import galaxea_zero_config
from lyc_changed.galaxea_policy_realworld_config_best import galaxea_zero_real_world_config

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Unnormalized key (e.g., "").
    unnorm_key: str
    
    norm_stats: str = '/EFM-Pretrain/galaxea_0/runs/pi0_libero/dataset_statistics.json'    


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    
def main(args: Args) -> None:
    policy = create_torch_policy(
        galaxea_zero_real_world_config, 
        args.policy.dir, 
        default_prompt=args.default_prompt,
        norm_stats=pathlib.Path(args.policy.dir) / "dataset_statistics.json",
        unnorm_key=args.policy.unnorm_key,
        sample_kwargs={
            'image_keys': ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
        }
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


# uv run lyc_changed/serve_policy.py policy:checkpoint --policy.config=pi0_libero --policy.dir=jax_params --policy.unnorm-key libero_spatial_no_noops
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
