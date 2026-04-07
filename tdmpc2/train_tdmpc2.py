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
    parser.add_argument("--plan-horizon", type=int, default=5, help="MPPI planning horizon.")
    parser.add_argument("--plan-samples", type=int, default=512, help="MPPI sample count.")
    parser.add_argument("--plan-temperature", type=float, default=0.5, help="MPPI temperature.")
    parser.add_argument("--use-sam", action="store_true", help="Enable SAM optimizer.")
    parser.add_argument("--sam-rho", type=float, default=0.05, help="SAM perturbation radius.")
    parser.add_argument(
        "--simnorm-dim",
        type=int,
        default=8,
        help="SimNorm group size (set <=0 to disable).",
    )
    parser.add_argument("--use-info-prop", action="store_true", help="Enable InfoProp planning.")
    parser.add_argument(
        "--info-prop-threshold",
        type=float,
        default=0.1,
        help="InfoProp uncertainty threshold.",
    )
    parser.add_argument(
        "--info-prop-ensemble-k",
        type=int,
        default=5,
        help="InfoProp MC-dropout samples.",
    )
    parser.add_argument(
        "--max-wall-clock-seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock budget for training (0 disables time limit).",
    )
    return parser


def main(
    env_name: str = "walker",
    task: str | None = None,
    run_name: str | None = None,
    total_steps: int = 10_000,
    dynamics_type: str = "mlp",
    ssm_state_dim: int = 256,
    plan_horizon: int = 5,
    plan_samples: int = 512,
    plan_temperature: float = 0.5,
    use_sam: bool = False,
    sam_rho: float = 0.05,
    simnorm_dim: int = 8,
    use_info_prop: bool = False,
    info_prop_threshold: float = 0.1,
    info_prop_ensemble_k: int = 5,
    max_wall_clock_seconds: float = 0.0,
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
    config_kwargs = {
        "total_steps": total_steps,
        "plan_horizon": plan_horizon,
        "plan_samples": plan_samples,
        "plan_temperature": plan_temperature,
        "use_sam": use_sam,
        "sam_rho": sam_rho,
        "use_info_prop": use_info_prop,
        "info_prop_threshold": info_prop_threshold,
        "info_prop_ensemble_k": info_prop_ensemble_k,
        "max_wall_clock_seconds": (
            max_wall_clock_seconds if max_wall_clock_seconds > 0 else None
        ),
    }
    if total_steps <= 1_000:
        config_kwargs.update(
            batch_size=32,
            seed_steps=min(100, total_steps),
            eval_every_steps=max(1, total_steps),
            log_every_steps=max(1, total_steps),
            rollout_error_every_steps=max(1, total_steps),
            n_eval_episodes=1,
        )
    if dynamics_type in {"s4", "s5", "mamba"}:
        config_kwargs.update(
            batch_size=min(config_kwargs.get("batch_size", 256), 128),
            plan_samples=min(plan_samples, 128),
            target_plan_samples=32,
        )
    config = TDMPC2TrainerConfig(**config_kwargs)
    model = TDMPC2Model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=config.latent_dim,
        dynamics_type=dynamics_type,
        ssm_state_dim=ssm_state_dim,
        simnorm_dim=simnorm_dim if simnorm_dim > 0 else None,
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
        plan_horizon=args.plan_horizon,
        plan_samples=args.plan_samples,
        plan_temperature=args.plan_temperature,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        simnorm_dim=args.simnorm_dim,
        use_info_prop=args.use_info_prop,
        info_prop_threshold=args.info_prop_threshold,
        info_prop_ensemble_k=args.info_prop_ensemble_k,
        max_wall_clock_seconds=args.max_wall_clock_seconds,
    )
