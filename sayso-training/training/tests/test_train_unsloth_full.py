from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_unsloth_full.py"
SPEC = importlib.util.spec_from_file_location("train_unsloth_full", SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def test_canary_keeps_the_full_run_schedule_and_stops_early() -> None:
    args = TRAIN.parse_args(["--canary", "--out", "/tmp/canary"])
    assert args.max_steps == -1
    assert args.stop_at_step == TRAIN.CANARY_STEPS == 250


def test_full_training_does_not_stop_early(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    record = tmp_path / "promoted.json"
    record.write_text(json.dumps({"promoted": True, "validation_passed": True, "canary_passed": True,
                                  "rendered_sha256": TRAIN.sha256(data)}))
    args = TRAIN.parse_args(["--data", str(data), "--promotion-record", str(record), "--out", "/tmp/full"])
    assert args.max_steps == -1
    assert args.stop_at_step == 0


def test_full_training_rejects_unpromoted_data(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    record = tmp_path / "promoted.json"
    record.write_text(json.dumps({"promoted": True, "validation_passed": True, "canary_passed": True,
                                  "rendered_sha256": "other"}))
    with pytest.raises(SystemExit):
        TRAIN.parse_args(["--data", str(data), "--promotion-record", str(record)])


def test_manifest_pins_model_files_data_and_args(tmp_path: Path) -> None:
    model = tmp_path / "model"
    (model / "sub").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "sub" / "tokenizer.json").write_text("tok")
    args = SimpleNamespace(canary=True, stop_at_step=250, model=str(model), data="/data.jsonl")
    manifest = TRAIN.build_manifest(args, {"seed": 1, "learning_rate": 5e-5}, "data-sha")
    assert manifest["mode"] == "canary"
    assert set(manifest["base_model_files"]) == {"config.json", "sub/tokenizer.json"}
    assert manifest["base_model_files"]["config.json"] == TRAIN.sha256(model / "config.json")
    assert manifest["data_sha256"] == "data-sha"
    assert manifest["training_args"]["seed"] == 1
    assert manifest["trainer_sha256"] == TRAIN.sha256(SCRIPT)
    assert "torch" in manifest["packages"]


def test_stop_callback_stops_at_the_requested_step() -> None:
    pytest.importorskip("transformers")
    callback = TRAIN.stop_callback(250)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_step_end(None, SimpleNamespace(global_step=249), control)
    assert control.should_training_stop is False
    callback.on_step_end(None, SimpleNamespace(global_step=250), control)
    assert control.should_training_stop is True
