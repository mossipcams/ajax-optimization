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


def _stub_docker(monkeypatch, sayso, returncode, stderr=""):
    commands = []

    def fake_run(command, *args, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(sayso.subprocess, "run", fake_run)
    return commands


def test_check_trainer_python_runs_the_stack_check_in_the_container(sayso, monkeypatch):
    commands = _stub_docker(monkeypatch, sayso, 0)
    sayso.check_trainer_python()
    [(command, kwargs)] = commands
    assert command[:5] == ["docker", "exec", "sayso-trainer", sayso.TRAINER_PYTHON, "-c"]
    assert "check_installed_stack" in command[5]
    assert f"{sayso.HOST_TRAINING}/training/scripts" in command[5]
    assert kwargs == {"check": False, "capture_output": True, "text": True}


def test_check_trainer_python_passes_when_interpreter_exists(sayso, monkeypatch):
    _stub_docker(monkeypatch, sayso, 0)
    sayso.check_trainer_python()


def test_check_trainer_python_exits_naming_interpreter_with_captured_output(sayso, monkeypatch):
    _stub_docker(monkeypatch, sayso, 1, stderr="boom")
    with pytest.raises(SystemExit) as excinfo:
        sayso.check_trainer_python()
    message = str(excinfo.value)
    assert sayso.TRAINER_PYTHON in message
    assert "boom" in message
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
    assert commands[0][0][:5] == ["docker", "exec", "sayso-trainer", sayso.TRAINER_PYTHON, "-c"]
    assert not (state / "train").exists()
    assert sayso.load(state / "A.json") == record
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def _stub_canary(monkeypatch, sayso, tmp_path, record, interpreter_check):
    calls = []

    def fake_render(rec):
        calls.append("render_and_stage")
        return "/data/staged.jsonl"

    monkeypatch.setattr(sayso, "current", lambda: record)
    monkeypatch.setattr(sayso, "digest", lambda path: "abc")
    monkeypatch.setattr(sayso, "ensure_vm", lambda: None)
    monkeypatch.setattr(sayso, "render_and_stage", fake_render)
    monkeypatch.setattr(sayso, "write_current", lambda rec: calls.append("write_current"))
    monkeypatch.setattr(sayso, "check_trainer_python", lambda: interpreter_check(calls))
    monkeypatch.setattr(sayso, "settle_gpu", lambda: None)
    monkeypatch.setattr(sayso, "run", lambda command, **kwargs: calls.append("train"))
    monkeypatch.setattr(sayso, "EVAL_RESULTS", tmp_path / "runs")
    _stub_docker(monkeypatch, sayso, 0)
    return calls


def _canary_record(tmp_path):
    return {"run_id": "r1", "validated": True, "validated_sha256": "abc",
            "dataset_path": str(tmp_path / "d.jsonl"), "rendered_path": str(tmp_path / "r.jsonl")}


def test_canary_refuses_a_missing_interpreter_before_staging_or_recording(sayso, monkeypatch, tmp_path):
    def missing(calls):
        calls.append("check_trainer_python")
        raise SystemExit("Trainer interpreter is missing")

    calls = _stub_canary(monkeypatch, sayso, tmp_path, _canary_record(tmp_path), missing)
    with pytest.raises(SystemExit):
        sayso.cmd_canary(argparse.Namespace(reuse_checkpoint=False))
    assert calls == ["check_trainer_python"]


def test_canary_reuse_does_not_check_the_trainer_interpreter(sayso, monkeypatch, tmp_path):
    (tmp_path / "runs" / "r1-canary" / "final").mkdir(parents=True)
    calls = _stub_canary(monkeypatch, sayso, tmp_path, _canary_record(tmp_path), lambda calls: calls.append("check_trainer_python"))
    sayso.cmd_canary(argparse.Namespace(reuse_checkpoint=True))
    assert "check_trainer_python" not in calls
    assert "train" not in calls
