import numpy as np
import matplotlib.pyplot as plt
import os
import json


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def load_sb3(path):
    data = np.load(path)
    return data["timesteps"], data["results"].mean(axis=1)


def load_tdmpc2(path):
    data = np.load(path)
    if hasattr(data, "files") and "timesteps" in data.files and "results" in data.files:
        timesteps = np.asarray(data["timesteps"])
        results = np.asarray(data["results"])
        if timesteps.size == 0 or results.size == 0:
            raise ValueError(f"No TD-MPC2 evaluation points found in {path}")
        return timesteps, results.mean(axis=1)
    return data[:, 0], data[:, 1]


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

    return np.asarray(timesteps), np.asarray(rewards)


def main():
    ppo_path = first_existing_path(
        "./logs/ppo_walker/eval/evaluations.npz",
        "./logs/ppo_walker_mac/eval/evaluations.npz",
    )
    sac_path = first_existing_path(
        "./logs/sac_cheetah/eval/evaluations.npz",
        "./logs/sac_cheetah_mac/eval/evaluations.npz",
    )
    tdmpc2_path = first_existing_path(
        "./logs/tdmpc2_walker_mlp/eval/evaluations.npz",
        "./logs/tdmpc2_walker/results.npy",
        "./logs/tdmpc2_walker_mac/results.npy",
    )
    tdmpc2_metrics_path = first_existing_path(
        "./artifacts/tdmpc2_walker_mlp/metrics.jsonl",
        "./artifacts/tdmpc2_walker/metrics.jsonl",
    )

    plots = []

    if ppo_path:
        plots.append(("PPO", "PPO — Walker Walk", "blue", load_sb3, ppo_path))

    if sac_path:
        plots.append(("SAC", "SAC — Cheetah Run", "orange", load_sb3, sac_path))

    if tdmpc2_path:
        plots.append(
            (
                "TD-MPC2 (MLP)",
                "TD-MPC2 — Walker Walk",
                "green",
                load_tdmpc2,
                tdmpc2_path,
            )
        )
    elif tdmpc2_metrics_path:
        plots.append(
            (
                "TD-MPC2 (MLP)",
                "TD-MPC2 — Walker Walk",
                "green",
                load_tdmpc2_metrics,
                tdmpc2_metrics_path,
            )
        )

    if not plots:
        raise FileNotFoundError("No evaluation files found under ./logs/")

    fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 5))
    if len(plots) == 1:
        axes = [axes]

    for index, (label, title, color, loader, path) in enumerate(plots):
        t, r = loader(path)
        axes[index].plot(t, r, color=color, label=label, marker="o")
        axes[index].set_title(title)
        axes[index].set_xlabel("Timesteps")
        axes[index].grid(True)
        axes[index].legend()

    axes[0].set_ylabel("Mean Reward")

    plt.tight_layout()
    plt.savefig("./logs/all_results.png", dpi=150)
    plt.show()
    print("Saved to ./logs/all_results.png")


if __name__ == "__main__":
    main()
