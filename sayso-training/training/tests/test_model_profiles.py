from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SMALL_230M = ["--model", "/workspace/host/lfm/models/LFM2.5-230M-Base", "--lr", "5e-5", "--warmup-ratio", "0.1",
              "--batch", "1", "--accum", "32"]
SMALL_350M = ["--model", "/workspace/host/lfm/models/LFM2.5-350M-Base-sayso", "--lr", "2e-5",
              "--warmup-ratio", "0.1", "--batch", "1", "--accum", "18", "--merge-views"]


@pytest.fixture()
def sayso(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("sayso_profiles", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setattr(module, "TRAINS", tmp_path / "state" / "train")
    monkeypatch.setattr(module, "EVAL_RESULTS", tmp_path / "runs")
    monkeypatch.setattr(module, "ensure_vm", lambda: None)
    monkeypatch.setattr(module, "check_trainer_python", lambda: None)
    monkeypatch.setattr(module, "check_trainable", lambda *args: None)
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: pytest.fail("trainer launched"))
    return module


def write_promotion(sayso, tmp_path, canary_profile):
    dataset = tmp_path / "A.jsonl"
    dataset.write_text("A")
    record = {"run_id": "A", "dataset_path": str(dataset), "dataset_sha256": sayso.digest(dataset),
              "rendered_path": str(dataset), "git_commit": "gen", "config": "cfg.yaml",
              "canary_profile": canary_profile}
    sayso.write_current(record)
    promotion = {"promoted": True, "validation_passed": True, "canary_passed": True, "run_id": "A",
                 "dataset_sha256": record["dataset_sha256"], "rendered_sha256": record["dataset_sha256"]}
    if canary_profile is not None:
        promotion["canary_profile"] = canary_profile
    sayso.save(sayso.STATE / "promoted_dataset.json", promotion)


def test_both_profiles_resolve_to_trainer_recipes(sayso):
    assert sayso.train_recipe("230m") == SMALL_230M
    assert sayso.train_recipe("350m") == SMALL_350M


def test_profile_trainer_model_maps_to_host_path(sayso):
    assert str(sayso.host_path(sayso.model_profile("350m")["trainer_model"])).endswith(
        "lfm/models/LFM2.5-350M-Base-sayso")


def test_unknown_profile_is_rejected_with_valid_names(sayso):
    with pytest.raises(SystemExit, match="valid profiles: 230m, 350m"):
        sayso.train_recipe("1b")


def test_train_refuses_when_model_differs_from_promoted_profile(sayso, tmp_path):
    write_promotion(sayso, tmp_path, "230m")
    with pytest.raises(SystemExit, match="does not match the promoted canary profile 230m"):
        sayso.cmd_train(argparse.Namespace(model="350m"))


def test_train_refuses_default_profile_when_promotion_was_350m(sayso, tmp_path):
    write_promotion(sayso, tmp_path, "350m")
    with pytest.raises(SystemExit, match="does not match the promoted canary profile 350m"):
        sayso.cmd_train(argparse.Namespace())


def test_train_refuses_promotion_without_canary_profile_for_350m(sayso, tmp_path):
    write_promotion(sayso, tmp_path, None)
    with pytest.raises(SystemExit, match="does not match the promoted canary profile 230m"):
        sayso.cmd_train(argparse.Namespace(model="350m"))
