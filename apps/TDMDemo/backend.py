"""ShipBench-only backend utilities for the TDM Streamlit demo."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "shipBench"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tdm_demo"
PYTHON_BIN = Path("/opt/miniconda3/envs/mesh/bin/python")
CHANNELS = ("U:0", "U:1", "U:2", "p_rgh")
MODEL_LABELS = {
    "app_transformer": "APP-Transformer",
    "fno": "FNO",
    "pcno": "PCNO",
}


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    conditions: tuple[str, ...]
    batch_size: int
    max_steps: int
    eval_every: int
    learning_rate: float
    train_ratio: float
    rollout_holdout_steps: int
    seed: int
    # APP
    patch_size: int = 256
    downsample_ratio: float = 0.6
    d_model: int = 56
    attention_heads: int = 4
    layers: int = 4
    # FNO / PCNO
    hidden_width: int = 32
    fourier_modes: int = 8
    neighbors: int = 8


def discover_conditions() -> dict[str, list[str]]:
    """Return locally available ShipBench hull/Reynolds conditions."""
    catalog: dict[str, list[str]] = {}
    if not DATA_ROOT.is_dir():
        return catalog
    for hull_dir in sorted(DATA_ROOT.iterdir()):
        field_dir = hull_dir / "field"
        if not field_dir.is_dir():
            continue
        conditions = [
            path.name
            for path in sorted(field_dir.iterdir())
            if path.is_dir() and (path / "flow_cache.npz").is_file()
        ]
        if conditions:
            catalog[hull_dir.name] = conditions
    return catalog


def condition_path(condition: str) -> Path:
    """Resolve a catalog key such as ``DTC/1Re`` to a checked local path."""
    hull, reynolds = condition.split("/", maxsplit=1)
    path = DATA_ROOT / hull / "field" / reynolds
    if not (path / "flow_cache.npz").is_file():
        raise FileNotFoundError(f"ShipBench cache not found: {path}")
    return path


def condition_summary(condition: str) -> dict[str, Any]:
    path = condition_path(condition)
    report_path = path / "flow_cache_alignment_report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata_path = path.parents[1] / "metadata.yaml"
    metadata = {}
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    return {
        "condition": condition,
        "frames": int(report.get("frames", 0)),
        "points": int(report.get("reference_points", 0)),
        "metadata": metadata,
    }


def checkpoint_dir(hull: str, model: str = "app_transformer") -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "natural_time_v2"
        / "joint_all_models_16k"
        / hull
        / model
        / "seed42"
    )


def available_checkpoints() -> list[dict[str, str]]:
    records = []
    for hull in discover_conditions():
        for model in MODEL_LABELS:
            run_dir = checkpoint_dir(hull, model)
            best = run_dir / "model_best_mae.pth"
            final = run_dir / "model_final.pth"
            checkpoint = best if best.is_file() else final
            if checkpoint.is_file():
                records.append(
                    {
                        "hull": hull,
                        "model": model,
                        "label": f"{hull} · {MODEL_LABELS[model]}",
                        "checkpoint": str(checkpoint),
                        "run_dir": str(run_dir),
                    }
                )
    return records


def load_training_metrics(hull: str, model: str) -> pd.DataFrame:
    path = checkpoint_dir(hull, model) / "metrics.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_rollout_metrics(hull: str, model: str) -> pd.DataFrame:
    path = checkpoint_dir(hull, model) / "rollout_metrics.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _relative_condition_path(condition: str) -> str:
    return str(condition_path(condition).relative_to(PROJECT_ROOT))


def build_training_command(config: TrainingConfig, run_dir: Path) -> list[str]:
    """Translate the UI configuration into an existing project training CLI."""
    if config.model not in MODEL_LABELS:
        raise ValueError(f"Unsupported demo model: {config.model}")
    if not config.conditions:
        raise ValueError("At least one ShipBench condition is required")

    data_dirs = [_relative_condition_path(item) for item in config.conditions]
    common = [
        str(PYTHON_BIN),
        str(
            PROJECT_ROOT
            / "scripts"
            / ("train_irregular_pcno.py" if config.model == "pcno" else (
                "train_patch.py" if config.model == "app_transformer" else "train_irregular.py"
            ))
        ),
        "--dataset_type",
        "ship",
        "--data_dirs",
        *data_dirs,
        "--batch_size",
        str(config.batch_size),
        "--max_steps",
        str(config.max_steps),
        "--eval_every",
        str(config.eval_every),
        "--log_every",
        str(min(50, config.eval_every)),
        "--lr",
        str(config.learning_rate),
        "--train_ratio",
        str(config.train_ratio),
        "--rollout_holdout_steps",
        str(config.rollout_holdout_steps),
        "--seed",
        str(config.seed),
        "--split_seed",
        str(config.seed),
        "--save_dir",
        str(run_dir),
        "--disable_tqdm",
    ]
    if len(config.conditions) > 1:
        common.append("--multi_condition")

    if config.model == "app_transformer":
        common.extend(
            [
                "--model",
                "transformer",
                "--patch_size",
                str(config.patch_size),
                "--partition_mode",
                "adaptive",
                "--downsample_method",
                "distance",
                "--downsample_ratio",
                str(config.downsample_ratio),
                "--d_model",
                str(config.d_model),
                "--nhead",
                str(config.attention_heads),
                "--num_layers",
                str(config.layers),
            ]
        )
    elif config.model == "fno":
        common.extend(
            [
                "--model",
                "fno",
                "--n_hidden",
                str(config.hidden_width),
                "--n_layers",
                str(config.layers),
                "--modes",
                str(config.fourier_modes),
            ]
        )
    else:
        common.extend(
            [
                "--layers",
                *([str(config.hidden_width)] * config.layers),
                "--n_modes",
                str(config.fourier_modes),
                "--k_neighbors",
                str(config.neighbors),
            ]
        )
    return common


def _job_file() -> Path:
    return OUTPUT_ROOT / "active_job.json"


def read_job() -> dict[str, Any] | None:
    path = _job_file()
    if not path.is_file():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    pid = int(job["pid"])
    try:
        finished_pid, _return_code = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        finished_pid = 0

    if finished_pid == pid:
        status = "finished"
    else:
        try:
            os.kill(pid, 0)
            status = "running"
        except (OSError, ProcessLookupError):
            status = "finished"
    job["status"] = status
    return job


def start_training(config: TrainingConfig) -> dict[str, Any]:
    active = read_job()
    if active and active["status"] == "running":
        raise RuntimeError(f"Training job {active['pid']} is already running")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"{stamp}_{config.model}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = build_training_command(config, run_dir)
    log_path = run_dir / "train.log"
    config_path = run_dir / "demo_config.json"
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_handle.close()
    job = {
        "pid": process.pid,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "command": command,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _job_file().write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return job


def stop_training() -> None:
    job = read_job()
    if not job or job["status"] != "running":
        return
    os.killpg(int(job["pid"]), signal.SIGTERM)


def tail_log(path: str | Path, lines: int = 80) -> str:
    log_path = Path(path)
    if not log_path.is_file():
        return "等待训练日志…"
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def result_to_csv(
    coords: np.ndarray, prediction: np.ndarray, target: np.ndarray
) -> bytes:
    frame = pd.DataFrame(
        np.column_stack([coords, prediction, target, prediction - target]),
        columns=[
            "x",
            "y",
            *[f"pred_{name}" for name in CHANNELS],
            *[f"true_{name}" for name in CHANNELS],
            *[f"error_{name}" for name in CHANNELS],
        ],
    )
    return frame.to_csv(index=False).encode("utf-8")
