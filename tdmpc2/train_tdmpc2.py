from __future__ import annotations

import argparse

from device_utils import describe_device, get_best_device
from env_setup import make_env
from run_layout import init_run_paths
from tdmpc2.model import TDMPC2Model
from tdmpc2.trainer import TDMPC2Trainer, TDMPC2TrainerConfig


TASK_DEFAULTS = {
    "walker": ("walk", "tdmpc2_walker_mlp"),
    "cheetah": ("run", "tdmpc2_cheetah_mlp"),
    "hopper": ("hop", "tdmpc2_hopper_mlp"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Phase 1 TD-MPC2 baseline with MLP dynamics."
    )
    parser.add_argument(
        "--env-name",
        choices=sorted(TASK_DEFAULTS),
        default="walker",
        help="DeepMind Control domain to train on.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Override the default task for the selected domain.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the output run directory name.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10_000,
        help="Number of environment steps to train for.",
    )
    parser.add_argument(
        "--dynamics-type",
        choices=["mlp", "s4", "s5", "mamba"],
        default="mlp",
        help="Dynamics module to use in the world model.",
    )
    parser.add_argument(
        "--ssm-state-dim",
        type=int,
        default=256,
        help="State dimension for SSM dynamics variants.",
    )
    return parser


def main(
    env_name: str = "walker",
    task: str | None = None,
    run_name: str | None = None,
    total_steps: int = 10_000,
    dynamics_type: str = "mlp",
    ssm_state_dim: int = 256,
) -> None:
    default_task, default_run_name = TASK_DEFAULTS[env_name]
    task = task or default_task
    run_name = run_name or default_run_name.replace("mlp", dynamics_type)

    paths = init_run_paths(run_name)
    device = get_best_device()
    print(f"Selected device: {describe_device()}")

    env = make_env(env_name, task)
    eval_env = make_env(env_name, task, seed=1)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    config = TDMPC2TrainerConfig(total_steps=total_steps)
    model = TDMPC2Model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=config.latent_dim,
        dynamics_type=dynamics_type,
        ssm_state_dim=ssm_state_dim,
    )

    trainer = TDMPC2Trainer(
        model=model,
        env=env,
        eval_env=eval_env,
        paths=paths,
        device=device,
        config=config,
        run_name=run_name,
        environment_name=f"{env_name}_{task}",
    )
    trainer.train()
    print(
        "Training output stored in "
        f"{paths.metrics_path} and {paths.summary_path}"
    )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(
        env_name=args.env_name,
        task=args.task,
        run_name=args.run_name,
        total_steps=args.total_steps,
        dynamics_type=args.dynamics_type,
        ssm_state_dim=args.ssm_state_dim,
    )
