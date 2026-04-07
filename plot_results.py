import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_sb3(path):
    data = np.load(path)
    return np.asarray(data["timesteps"]), np.asarray(data["results"]).mean(axis=1)


def _ensure_2d_results(results: np.ndarray) -> np.ndarray:
    if results.ndim == 1:
        return results[:, None]
    return results


def _sort_xy(timesteps: np.ndarray, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(timesteps)
    return timesteps[order], rewards[order]


def load_tdmpc2(path):
    data = np.load(path)
    if hasattr(data, "files") and "timesteps" in data.files and "results" in data.files:
        timesteps = np.asarray(data["timesteps"])
        results = _ensure_2d_results(np.asarray(data["results"]))
        if timesteps.size == 0 or results.size == 0:
            raise ValueError(f"No TD-MPC2 evaluation points found in {path}")
        mean_rewards = results.mean(axis=1)
        return _sort_xy(timesteps, mean_rewards)

    legacy = np.asarray(data)
    if legacy.ndim == 1 and legacy.size >= 2:
        legacy = legacy.reshape(-1, 2)
    if legacy.ndim != 2 or legacy.shape[1] < 2:
        raise ValueError(f"Unsupported TD-MPC2 result format in {path}")
    return _sort_xy(legacy[:, 0], legacy[:, 1])


def load_tdmpc2_metrics(path):
    timesteps = []
    rewards = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "eval_reward_mean" in record:
                timesteps.append(int(record["step"]))
                rewards.append(float(record["eval_reward_mean"]))

    if not timesteps:
        raise ValueError(f"No eval_reward_mean entries found in {path}")

    return _sort_xy(np.asarray(timesteps), np.asarray(rewards))


def _find_tdmpc_variant_path(variant: str) -> tuple[str, str] | None:
    preferred_npz = [
        f"./logs/tdmpc2_walker_{variant}/eval/evaluations.npz",
        f"./logs/tdmpc2_walker_{variant}_us/eval/evaluations.npz",
    ]
    preferred_metrics = [
        f"./artifacts/tdmpc2_walker_{variant}/metrics.jsonl",
        f"./artifacts/tdmpc2_walker_{variant}_us/metrics.jsonl",
    ]

    npz_path = first_existing_path(*preferred_npz)
    if npz_path:
        return npz_path, "npz"

    metrics_path = first_existing_path(*preferred_metrics)
    if metrics_path:
        return metrics_path, "metrics"

    wildcard_npz = sorted(glob.glob(f"./logs/tdmpc2_walker*{variant}*/eval/evaluations.npz"))
    if wildcard_npz:
        return wildcard_npz[0], "npz"

    wildcard_metrics = sorted(glob.glob(f"./artifacts/tdmpc2_walker*{variant}*/metrics.jsonl"))
    if wildcard_metrics:
        return wildcard_metrics[0], "metrics"

    return None


def main():
    ppo_path = first_existing_path(
        "./logs/ppo_walker/eval/evaluations.npz",
        "./logs/ppo_walker_mac/eval/evaluations.npz",
        "./logs/ppo_walker_us/eval/evaluations.npz",
    )
    sac_path = first_existing_path(
        "./logs/sac_cheetah/eval/evaluations.npz",
        "./logs/sac_cheetah_mac/eval/evaluations.npz",
        "./logs/sac_cheetah_us/eval/evaluations.npz",
    )

    plots = []

    if ppo_path:
        plots.append(("PPO", "PPO — Walker Walk", "blue", load_sb3, ppo_path))

    if sac_path:
        plots.append(("SAC", "SAC — Cheetah Run", "orange", load_sb3, sac_path))

    tdmpc_variants = [
        ("mlp", "TD-MPC2 (MLP)", "green"),
        ("s4", "TD-MPC2 (S4)", "purple"),
        ("s5", "TD-MPC2 (S5)", "red"),
        ("mamba", "TD-MPC2 (Mamba)", "brown"),
    ]

    for variant_key, variant_label, variant_color in tdmpc_variants:
        path_info = _find_tdmpc_variant_path(variant_key)
        if not path_info:
            continue

        variant_path, source_kind = path_info
        loader = load_tdmpc2 if source_kind == "npz" else load_tdmpc2_metrics
        plots.append(
            (
                variant_label,
                f"{variant_label} — Walker Walk",
                variant_color,
                loader,
                variant_path,
            )
        )

    if not plots:
        raise FileNotFoundError("No evaluation files found under ./logs/ or ./artifacts/")

    fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 5))
    if len(plots) == 1:
        axes = [axes]

    for index, (label, title, color, loader, path) in enumerate(plots):
        t, r = loader(path)
        marker = "o" if len(t) == 1 else None
        axes[index].plot(t, r, color=color, label=label, linewidth=2.0, marker=marker)
        axes[index].set_title(title)
        axes[index].set_xlabel("Timesteps")
        axes[index].grid(True, alpha=0.3)
        axes[index].legend()

    axes[0].set_ylabel("Mean Reward")

    output_path = Path("./logs/all_results.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
