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
    module = load_sayso()
    monkeypatch.setattr(module, "EVAL_RESULTS", Path("/unused-eval-results"))
    return module


class RunCapture:
    def __init__(self):
        self.docker_commands = []

    def fake_subprocess_run(self, command, *args, **kwargs):
        self.docker_commands.append(command)
        return subprocess.CompletedProcess(command, 0)


def test_default_host_training(sayso):
    assert sayso.HOST_TRAINING == "/workspace/host/lfm/ajax-optimization/sayso-training"


def test_env_override_host_training(monkeypatch):
    monkeypatch.setenv("SAYSO_TRAINING_HOST_ROOT", "/custom/training")
    module = load_sayso()
    assert module.HOST_TRAINING == "/custom/training"


def test_canary_eval_passes_profile(sayso, monkeypatch):
    capture = RunCapture()
    monkeypatch.setattr(sayso.subprocess, "run", capture.fake_subprocess_run)
    passed, _ = sayso.eval_checkpoint("/checkpoint/final", "out", "smoke", "canary")
    assert passed
    command = capture.docker_commands[0]
    assert "--profile" in command
    assert command[command.index("--profile") + 1] == "canary"


def test_canary_eval_omits_profile_by_default(sayso, monkeypatch):
    capture = RunCapture()
    monkeypatch.setattr(sayso.subprocess, "run", capture.fake_subprocess_run)
    sayso.eval_checkpoint("/checkpoint/final", "out", "smoke")
    assert "--profile" not in capture.docker_commands[0]


def _prepare_canary_module(sayso, monkeypatch, tmp_path, record):
    monkeypatch.setattr(sayso, "current", lambda: record)
    monkeypatch.setattr(sayso, "digest", lambda path: "abc")
    monkeypatch.setattr(sayso, "ensure_vm", lambda: None)
    monkeypatch.setattr(sayso, "render_and_stage", lambda rec: "/data/staged.jsonl")
    monkeypatch.setattr(sayso, "write_current", lambda rec: None)
    monkeypatch.setattr(sayso, "settle_gpu", lambda: None)
    train_commands = []
    monkeypatch.setattr(sayso, "run", lambda command, **kwargs: train_commands.append(command))
    capture = RunCapture()
    monkeypatch.setattr(sayso.subprocess, "run", capture.fake_subprocess_run)
    monkeypatch.setattr(sayso, "EVAL_RESULTS", tmp_path / "runs")
    return train_commands, capture


def test_canary_trains_and_evals_with_profile(sayso, monkeypatch, tmp_path):
    record = {"run_id": "r1", "validated": True, "validated_sha256": "abc",
              "dataset_path": str(tmp_path / "d.jsonl"), "rendered_path": str(tmp_path / "r.jsonl")}
    train_commands, capture = _prepare_canary_module(sayso, monkeypatch, tmp_path, record)
    sayso.cmd_canary(type("Args", (), {"reuse_checkpoint": False})())
    assert len(train_commands) == 1
    assert "--canary" in train_commands[0]
    command = capture.docker_commands[0]
    assert command[command.index("--profile") + 1] == "canary"


def test_canary_reuse_skips_training(sayso, monkeypatch, tmp_path):
    record = {"run_id": "r1", "validated": True, "validated_sha256": "abc",
              "dataset_path": str(tmp_path / "d.jsonl"), "rendered_path": str(tmp_path / "r.jsonl")}
    (tmp_path / "runs" / "r1-canary" / "final").mkdir(parents=True)
    train_commands, capture = _prepare_canary_module(sayso, monkeypatch, tmp_path, record)
    sayso.cmd_canary(type("Args", (), {"reuse_checkpoint": True})())
    assert train_commands == []
    command = capture.docker_commands[0]
    assert any("r1-canary/final" in str(part) for part in command)


def test_canary_reuse_without_checkpoint_fails(sayso, monkeypatch, tmp_path):
    record = {"run_id": "r1", "validated": True, "validated_sha256": "abc",
              "dataset_path": str(tmp_path / "d.jsonl"), "rendered_path": str(tmp_path / "r.jsonl")}
    _prepare_canary_module(sayso, monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="No canary checkpoint to reuse"):
        sayso.cmd_canary(type("Args", (), {"reuse_checkpoint": True})())


def test_main_parses_reuse_checkpoint_flag(sayso, monkeypatch):
    seen = {}

    def fake_cmd_canary(args):
        seen["flag"] = args.reuse_checkpoint

    monkeypatch.setattr(sayso, "cmd_canary", fake_cmd_canary)
    monkeypatch.setattr(sayso.sys, "argv", ["sayso", "canary", "--reuse-checkpoint"])
    assert sayso.main() == 0
    assert seen["flag"] is True
