import argparse
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
            step = record.get("timesteps", record.get("step"))
            metrics = record.get("metrics", {})
            reward = metrics.get("eval/mean_reward", record.get("eval_reward_mean"))
            if step is not None and reward is not None:
                timesteps.append(int(step))
                rewards.append(float(reward))

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


def _load_tdmpc_run(run_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    npz_path = Path(f"./logs/{run_name}/eval/evaluations.npz")
    metrics_path = Path(f"./artifacts/{run_name}/metrics.jsonl")
    if npz_path.exists():
        return load_tdmpc2(str(npz_path))
    if metrics_path.exists():
        return load_tdmpc2_metrics(str(metrics_path))
    return None


def _save_plot(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def _plot_single_case(run_name: str, label: str, output_dir: Path) -> bool:
    values = _load_tdmpc_run(run_name)
    if values is None:
        return False
    t, r = values
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    marker = "o" if len(t) == 1 else None
    ax.plot(t, r, linewidth=2.0, marker=marker)
    ax.set_title(label)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / f"{run_name}.png")
    plt.close(fig)
    return True


def plot_phase3_horizon_ablation(output_dir: Path | None = None) -> None:
    output_dir = output_dir or Path("./artifacts/plots/phase3")
    runs = [
        ("tdmpc2_walker_mlp_h5", "MLP H=5", "tab:green"),
        ("tdmpc2_walker_mlp_h10", "MLP H=10", "tab:olive"),
        ("tdmpc2_walker_s5_h5", "S5 H=5", "tab:red"),
        ("tdmpc2_walker_s5_h10", "S5 H=10", "tab:purple"),
    ]

    plotted = []
    for run_name, label, color in runs:
        values = _load_tdmpc_run(run_name)
        if values is None:
            continue
        plotted.append((label, color, *values))

    if not plotted:
        raise FileNotFoundError(
            "No Phase 3 runs found. Expected run names: "
            "tdmpc2_walker_mlp_h5, tdmpc2_walker_mlp_h10, tdmpc2_walker_s5_h5, tdmpc2_walker_s5_h10."
        )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label, color, t, r in plotted:
        marker = "o" if len(t) == 1 else None
        ax.plot(t, r, label=label, color=color, linewidth=2.0, marker=marker)
    ax.set_title("Phase 3 Planning Stability: Horizon Ablation (Walker-Walk)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_plot(fig, output_dir / "phase3_horizon_ablation.png")
    plt.close(fig)


def plot_overview():
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

    output_path = Path("./artifacts/plots/overview/all_results.png")
    _save_plot(fig, output_path)
    plt.close(fig)


def plot_phase0_baselines(output_dir: Path) -> None:
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
    entries = []
    if ppo_path:
        entries.append(("PPO Walker", *load_sb3(ppo_path)))
    if sac_path:
        entries.append(("SAC Cheetah", *load_sb3(sac_path)))
    if not entries:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for label, t, r in entries:
        ax.plot(t, r, linewidth=2.0, label=label, marker="o" if len(t) == 1 else None)
    ax.set_title("Phase 0 Baselines")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_plot(fig, output_dir / "phase0_baselines.png")
    plt.close(fig)


def plot_phase1_mlp(output_dir: Path) -> None:
    _plot_single_case("tdmpc2_walker_mlp", "Phase 1 - TD-MPC2 MLP (Walker)", output_dir)


def plot_phase2_ssm(output_dir: Path) -> None:
    runs = [
        ("tdmpc2_walker_mlp", "MLP", "tab:green"),
        ("tdmpc2_walker_s4", "S4", "tab:blue"),
        ("tdmpc2_walker_s5", "S5", "tab:red"),
        ("tdmpc2_walker_mamba", "Mamba", "tab:purple"),
    ]
    plotted = []
    for run_name, label, color in runs:
        values = _load_tdmpc_run(run_name)
        if values is not None:
            plotted.append((label, color, *values))
    if not plotted:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label, color, t, r in plotted:
        ax.plot(t, r, linewidth=2.0, color=color, label=label, marker="o" if len(t) == 1 else None)
    ax.set_title("Phase 2 Dynamics Comparison")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_plot(fig, output_dir / "phase2_dynamics_comparison.png")
    plt.close(fig)


def plot_all_phases() -> None:
    base_dir = Path("./artifacts/plots")
    plot_overview()
    plot_phase0_baselines(base_dir / "phase0")
    plot_phase1_mlp(base_dir / "phase1")
    plot_phase2_ssm(base_dir / "phase2")
    try:
        plot_phase3_horizon_ablation(base_dir / "phase3")
    except FileNotFoundError:
        pass
    # Save individual plots for all common Phase 1-3 run names if available.
    for run_name in [
        "tdmpc2_walker_mlp",
        "tdmpc2_walker_s4",
        "tdmpc2_walker_s5",
        "tdmpc2_walker_mamba",
        "tdmpc2_walker_mlp_h5",
        "tdmpc2_walker_mlp_h10",
        "tdmpc2_walker_s5_h5",
        "tdmpc2_walker_s5_h10",
    ]:
        _plot_single_case(run_name, run_name, base_dir / "all_cases")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot RL training results.")
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Plot only the Phase 3 horizon ablation (H=5 vs H=10, MLP vs S5).",
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Generate plots for Phase 0/1/2/3 and all common run cases under artifacts/plots.",
    )
    args = parser.parse_args(argv)

    if args.all_phases:
        plot_all_phases()
        return
    if args.phase3:
        plot_phase3_horizon_ablation()
        return
    plot_overview()


if __name__ == "__main__":
    main()
