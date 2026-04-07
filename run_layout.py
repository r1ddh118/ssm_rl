from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    log_dir: Path
    best_dir: Path
    eval_dir: Path
    artifact_dir: Path
    metrics_path: Path
    summary_path: Path
    model_path: Path
    eval_npz_path: Path


def init_run_paths(run_name: str) -> RunPaths:
    log_dir = Path("logs") / run_name
    artifact_dir = Path("artifacts") / run_name
    best_dir = log_dir / "best"
    eval_dir = log_dir / "eval"

    for path in (log_dir, artifact_dir):
        if path.exists():
            shutil.rmtree(path)

    for path in (log_dir, best_dir, eval_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_name=run_name,
        log_dir=log_dir,
        best_dir=best_dir,
        eval_dir=eval_dir,
        artifact_dir=artifact_dir,
        metrics_path=artifact_dir / "metrics.jsonl",
        summary_path=artifact_dir / "summary.json",
        model_path=log_dir / "final_model",
        eval_npz_path=eval_dir / "evaluations.npz",
    )
