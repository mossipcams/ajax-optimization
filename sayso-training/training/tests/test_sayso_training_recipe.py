import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_sayso():
    loader = importlib.machinery.SourceFileLoader("sayso_cli", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader("sayso_cli", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def sayso(monkeypatch):
    monkeypatch.delenv("SAYSO_TRAINING_HOST_ROOT", raising=False)
    monkeypatch.delenv("TRAIN_MODEL", raising=False)
    monkeypatch.delenv("TRAIN_LR", raising=False)
    monkeypatch.delenv("TRAIN_WARMUP_RATIO", raising=False)
    monkeypatch.setenv("BASE_MODEL", "/models/LFM2.5-230M-Base")
    return load_sayso()


def test_trainer_args_unchanged_when_env_unset(sayso):
    assert sayso.trainer_args() == ["--lr", "2e-5"]


def test_trainer_args_pass_through_recipe_env(sayso, monkeypatch):
    monkeypatch.setenv("TRAIN_MODEL", "/srv/llm/lfm/models/LFM2.5-230M")
    monkeypatch.setenv("TRAIN_LR", "5e-5")
    monkeypatch.setenv("TRAIN_WARMUP_RATIO", "0.10")
    assert sayso.trainer_args() == [
        "--model", "/srv/llm/lfm/models/LFM2.5-230M",
        "--lr", "5e-5",
        "--warmup-ratio", "0.1",
    ]


def test_record_training_fields_with_env_model(sayso, monkeypatch, tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    expected_sha = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setenv("TRAIN_MODEL", str(tmp_path))
    monkeypatch.setenv("TRAIN_LR", "5e-5")
    record: dict = {}
    sayso.record_training_fields(record)
    assert record["model_path"] == str(tmp_path)
    assert record["model_sha256"] == expected_sha
    assert record["learning_rate"] == 5e-5


def test_record_training_fields_omits_model_when_env_unset(sayso):
    record: dict = {}
    sayso.record_training_fields(record)
    assert "model_path" not in record
    assert "model_sha256" not in record
    assert record["learning_rate"] == 2e-5
