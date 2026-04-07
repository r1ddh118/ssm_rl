import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from artifact_logging import JsonLinesMetricCallback, utc_now_iso
from device_utils import describe_device, get_best_device
from env_setup import make_env
from run_layout import init_run_paths


def main():
    run_name = "ppo_walker"
    device = get_best_device()
    device_description = describe_device()
    paths = init_run_paths(run_name)

    env = make_env("walker", "walk")
    eval_env = make_env("walker", "walk")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=0,
        device=device,
        tensorboard_log=str(paths.log_dir),
    )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=str(paths.best_dir),
            log_path=str(paths.eval_dir),
            eval_freq=10000,
            verbose=0,
        ),
        JsonLinesMetricCallback(str(paths.metrics_path), log_every_steps=10_000),
    ])

    started_at = utc_now_iso()
    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(paths.model_path))

    summary = {
        "run_name": run_name,
        "algorithm": "PPO",
        "environment": "walker_walk",
        "device": device,
        "device_description": device_description,
        "total_timesteps": total_timesteps,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now_iso(),
        "artifacts": {
            "metrics_jsonl": str(paths.metrics_path),
            "summary_json": str(paths.summary_path),
            "model": str(paths.model_path),
            "eval_npz": str(paths.eval_npz_path),
        },
    }
    with open(paths.summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "Training output stored in "
        f"{paths.metrics_path} and {paths.summary_path}"
    )


if __name__ == "__main__":
    main()
