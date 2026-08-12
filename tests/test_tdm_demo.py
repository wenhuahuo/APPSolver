from pathlib import Path

import numpy as np
import pytest

import apps.TDMDemo.backend as backend
from apps.TDMDemo.backend import (
    TrainingConfig,
    build_training_command,
    condition_summary,
    discover_conditions,
    read_job,
    result_to_csv,
)


def make_config(model: str) -> TrainingConfig:
    return TrainingConfig(
        model=model,
        conditions=("DTC/1Re", "DTC/2Re"),
        batch_size=2,
        max_steps=100,
        eval_every=20,
        learning_rate=1e-4,
        train_ratio=0.8,
        rollout_holdout_steps=20,
        seed=42,
    )


def test_shipbench_catalog_and_summary():
    catalog = discover_conditions()
    if not catalog:
        pytest.skip("local ShipBench data is not available")
    assert set(catalog) >= {"DTC", "KCS", "KVLCC2"}
    summary = condition_summary("DTC/1Re")
    assert summary["frames"] == 350
    assert summary["points"] > 0


@pytest.mark.parametrize(
    ("model", "script", "model_flag"),
    [
        ("app_transformer", "train_patch.py", "transformer"),
        ("fno", "train_irregular.py", "fno"),
        ("pcno", "train_irregular_pcno.py", None),
    ],
)
def test_training_command_uses_existing_ship_workflows(model, script, model_flag):
    command = build_training_command(make_config(model), Path("outputs/tdm_demo/test"))
    assert script in command[1]
    assert command[command.index("--dataset_type") + 1] == "ship"
    assert "--multi_condition" in command
    assert "cfd" not in " ".join(command).lower()
    if model_flag is not None:
        assert command[command.index("--model") + 1] == model_flag


def test_read_job_reaps_finished_child(monkeypatch, tmp_path):
    job_file = tmp_path / "active_job.json"
    job_file.write_text('{"pid": 123, "status": "running"}', encoding="utf-8")
    monkeypatch.setattr(backend, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(backend.os, "waitpid", lambda pid, options: (pid, 0))

    assert read_job()["status"] == "finished"


def test_read_job_keeps_live_child_running(monkeypatch, tmp_path):
    job_file = tmp_path / "active_job.json"
    job_file.write_text('{"pid": 123, "status": "running"}', encoding="utf-8")
    monkeypatch.setattr(backend, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(backend.os, "waitpid", lambda pid, options: (0, 0))
    monkeypatch.setattr(backend.os, "kill", lambda pid, signal: None)

    assert read_job()["status"] == "running"


def test_prediction_csv_contains_prediction_target_and_error():
    coords = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    prediction = np.ones((2, 4), dtype=np.float32)
    target = np.zeros((2, 4), dtype=np.float32)
    text = result_to_csv(coords, prediction, target).decode("utf-8")
    assert "pred_U:0" in text
    assert "true_p_rgh" in text
    assert "error_U:2" in text
