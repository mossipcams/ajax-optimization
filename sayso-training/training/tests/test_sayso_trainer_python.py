import argparse
import importlib.machinery
import importlib.util
import subprocess
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
    return load_sayso()


def _stub_docker(monkeypatch, sayso, returncode):
    commands = []

    def fake_run(command, *args, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(sayso.subprocess, "run", fake_run)
    return commands


def test_check_trainer_python_probes_the_interpreter_in_the_container(sayso, monkeypatch):
    commands = _stub_docker(monkeypatch, sayso, 0)
    sayso.check_trainer_python()
    assert commands == [(["docker", "exec", "unsloth", "test", "-x", sayso.TRAINER_PYTHON],
                         {"check": False})]


def test_check_trainer_python_passes_when_interpreter_exists(sayso, monkeypatch):
    _stub_docker(monkeypatch, sayso, 0)
    sayso.check_trainer_python()


def test_check_trainer_python_exits_naming_interpreter_when_missing(sayso, monkeypatch):
    _stub_docker(monkeypatch, sayso, 1)
    with pytest.raises(SystemExit) as excinfo:
        sayso.check_trainer_python()
    message = str(excinfo.value)
    assert sayso.TRAINER_PYTHON in message
    assert "scripts/llm-host/sayso-trainer-venv.sh" in message


def test_train_refuses_a_missing_interpreter_before_writing_any_record(sayso, tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setattr(sayso, "STATE", state)
    monkeypatch.setattr(sayso, "TRAINS", state / "train")
    monkeypatch.setattr(sayso, "EVAL_RESULTS", tmp_path / "runs")
    for name in ("ensure_vm", "gpu", "settle_gpu"):
        monkeypatch.setattr(sayso, name, lambda *args: None)
    monkeypatch.setattr(sayso, "ensure_pinned", lambda: {"git_commit": "c0de", "sayso_commit": "5a75"})
    monkeypatch.setattr(sayso, "render_and_stage", lambda record: "/staged.jsonl")
    monkeypatch.setattr(sayso, "run", lambda command, **kwargs: None)
    commands = _stub_docker(monkeypatch, sayso, 1)
    dataset = tmp_path / "A.jsonl"
    dataset.write_text("A")
    record = {"run_id": "A", "dataset_path": str(dataset), "dataset_sha256": sayso.digest(dataset),
              "rendered_path": str(dataset), "git_commit": "gen", "config": "cfg.yaml", "trained": False}
    sayso.save(state / "A.json", record)
    sayso.save(state / "current.json", {"run_id": "A"})
    sayso.save(state / "promoted_dataset.json",
               {"promoted": True, "validation_passed": True, "canary_passed": True, "run_id": "A",
                "dataset_sha256": record["dataset_sha256"], "rendered_sha256": record["dataset_sha256"]})
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as excinfo:
        sayso.cmd_train(argparse.Namespace())
    assert sayso.TRAINER_PYTHON in str(excinfo.value)
    assert commands[0][0] == ["docker", "exec", "unsloth", "test", "-x", sayso.TRAINER_PYTHON]
    assert not (state / "train").exists()
    assert sayso.load(state / "A.json") == record
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
