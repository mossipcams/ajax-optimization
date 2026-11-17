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
