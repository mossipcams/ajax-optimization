from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def sayso(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("sayso_auto_eval", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setattr(module, "TRAINS", tmp_path / "state" / "train")
    monkeypatch.setattr(module, "EVAL_RESULTS", tmp_path / "runs")
    monkeypatch.setattr(module, "safety_flags", lambda: {"unsafe"})
    for name in ("ensure_vm", "gpu", "settle_gpu", "check_trainer_python"):
        monkeypatch.setattr(module, name, lambda *args: None)
    monkeypatch.setattr(module, "ensure_pinned", lambda: {"git_commit": "c0de", "sayso_commit": "5a75"})
    monkeypatch.setattr(module, "render_and_stage", lambda record: "/staged.jsonl")
    monkeypatch.setattr(module, "run", lambda command, **kwargs: None)
    return module


def args(**values):
    return argparse.Namespace(**{"train_id": None, "manual_override": False, "reason": None, **values})


def trained_run(sayso, checkpoints, max_steps=2500):
    training = {"train_id": "T", "dataset_run_id": "A", "profile": "350m", "trained": True,
                "output": "/workspace/host/lfm/runs/sayso-promotion/T-full",
                "manifest": {"max_steps": max_steps} if max_steps else None}
    sayso.save(sayso.STATE / "A.json", {"run_id": "A", "staged_eval_data": "/workspace/host/datasets/A.val.jsonl"})
    sayso.save_training(training)
    for name in checkpoints:
        (sayso.local_output(training) / name).mkdir(parents=True)
    return training


def stub_eval(monkeypatch, sayso, promotion, heldout):
    calls = []

    def fake_eval(checkpoint, out_name, suite, profile=None):
        name = Path(checkpoint).name
        calls.append((suite, name))
        spec = promotion[name]
        summary = {"overall_pass_rate": spec["rate"], "passed": round(spec["rate"] * 10), "total": 10,
                   "flags": spec.get("flags", {})}
        return True, {"promotion": {"summary": summary, "result_dir": str(sayso.EVAL_RESULTS / out_name)}}

    def fake_heldout(training, checkpoint, name):
        calls.append(("heldout", name))
        return {"passed": round(heldout[name] * 300), "total": 300, "pass_rate": heldout[name],
                "result_dir": str(sayso.EVAL_RESULTS / f"{name}-heldout")}

    monkeypatch.setattr(sayso, "eval_checkpoint", fake_eval)
    monkeypatch.setattr(sayso, "eval_heldout", fake_heldout)
    return calls


def test_mid_checkpoint_is_the_one_nearest_57_percent(sayso, monkeypatch):
    names = ["checkpoint-200", "checkpoint-1000", "checkpoint-1400", "checkpoint-1800"]
    training = trained_run(sayso, names)
    promotion = {name: {"rate": 0.5} for name in ("checkpoint-1400", "final")}
    calls = stub_eval(monkeypatch, sayso, promotion, {name: 0.5 for name in promotion})
    sayso.auto_eval(training)
    assert {name for _, name in calls} == {"checkpoint-1400", "final"}
    assert set(sayso.training_run(args())["auto_eval"]) == {"checkpoint-1400", "final"}


def test_mid_checkpoint_falls_back_to_highest_step_without_manifest(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-200", "checkpoint-1000", "checkpoint-1800"], max_steps=None)
    promotion = {name: {"rate": 0.5} for name in ("checkpoint-1800", "final")}
    stub_eval(monkeypatch, sayso, promotion, {name: 0.5 for name in promotion})
    sayso.auto_eval(training)
    assert set(sayso.training_run(args())["auto_eval"]) == {"checkpoint-1800", "final"}


@pytest.mark.parametrize(("promotion", "heldout", "expected"), [
    ({"checkpoint-1400": {"rate": 0.95, "flags": {"unsafe": 1}}, "final": {"rate": 0.6}},
     {"checkpoint-1400": 0.99, "final": 0.1}, "final"),
    ({"checkpoint-1400": {"rate": 0.8}, "final": {"rate": 0.7}},
     {"checkpoint-1400": 0.2, "final": 0.9}, "checkpoint-1400"),
    ({"checkpoint-1400": {"rate": 0.8}, "final": {"rate": 0.8}},
     {"checkpoint-1400": 0.5, "final": 0.6}, "final"),
    ({"checkpoint-1400": {"rate": 0.8}, "final": {"rate": 0.8}},
     {"checkpoint-1400": 0.5, "final": 0.5}, "checkpoint-1400"),
])
def test_winner_is_safety_then_promotion_then_heldout(sayso, monkeypatch, promotion, heldout, expected):
    training = trained_run(sayso, ["checkpoint-1400"])
    stub_eval(monkeypatch, sayso, promotion, heldout)
    sayso.auto_eval(training)
    assert sayso.training_run(args())["selected_checkpoint"].endswith(f"/{expected}")


def test_auto_eval_records_table_selection_and_resets_release(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])
    training.update(release={"gguf": "old"}, final_metrics={"old": True}, final_eval_passed=True)
    sayso.save_training(training)
    promotion = {"checkpoint-1400": {"rate": 0.8, "flags": {"unsafe": 0}}, "final": {"rate": 0.7}}
    stub_eval(monkeypatch, sayso, promotion, {"checkpoint-1400": 0.4, "final": 0.9})
    sayso.auto_eval(training)
    record = sayso.training_run(args())
    assert record["auto_eval"]["checkpoint-1400"] == {
        "promotion_pass_rate": 0.8, "promotion_passed": 8, "promotion_total": 10, "safety_flags": 0,
        "heldout_pass_rate": 0.4, "heldout_total": 300,
        "result_dirs": {"promotion": str(sayso.EVAL_RESULTS / "T-auto-checkpoint-1400"),
                        "heldout": str(sayso.EVAL_RESULTS / "checkpoint-1400-heldout")}}
    assert record["selected_checkpoint"].endswith("/checkpoint-1400")
    assert record["release"] is None and record["final_metrics"] is None and record["final_eval_passed"] is False
    assert record["trained"] is True


def test_report_json_is_written_with_flags_per_checkpoint(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])
    promotion = {"checkpoint-1400": {"rate": 0.8, "flags": {"unsafe": 0, "wrong_domain": 2}}, "final": {"rate": 0.7}}
    stub_eval(monkeypatch, sayso, promotion, {"checkpoint-1400": 0.4, "final": 0.9})
    sayso.auto_eval(training)
    report = json.loads((sayso.local_output(training) / "report.json").read_text(encoding="utf-8"))
    assert report["train_id"] == "T"
    assert report["profile"] == "350m"
    assert report["dataset_run_id"] == "A"
    assert report["selected_checkpoint"] == f"{training['output']}/checkpoint-1400"
    assert set(report["auto_eval"]) == {"checkpoint-1400", "final"}
    assert report["checkpoints"]["checkpoint-1400"]["promotion_flags"] == {"unsafe": 0, "wrong_domain": 2}
    assert report["checkpoints"]["final"]["promotion_flags"] == {}


def test_auto_eval_failure_is_recorded_and_reraised_with_trained_kept(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])

    def broken(checkpoint, out_name, suite, profile=None):
        raise SystemExit("boom")

    monkeypatch.setattr(sayso, "eval_checkpoint", broken)
    with pytest.raises(SystemExit, match="boom"):
        sayso.auto_eval(training)
    record = sayso.training_run(args())
    assert record["failure"] == {"stage": "auto_eval", "reason": "boom"}
    assert record["trained"] is True


def test_select_resumes_auto_eval_for_the_trained_run(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])
    training.update(failure={"stage": "auto_eval", "reason": "boom"})
    sayso.save_training(training)
    calls = stub_eval(monkeypatch, sayso, {"checkpoint-1400": {"rate": 0.8}, "final": {"rate": 0.7}},
                      {"checkpoint-1400": 0.4, "final": 0.9})
    sayso.cmd_select(args())
    assert ("promotion", "final") in calls and ("heldout", "final") in calls
    record = sayso.training_run(args())
    assert record["selected_checkpoint"].endswith("/checkpoint-1400")
    assert record["failure"] is None


def test_select_refuses_an_untrained_run(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])
    training["trained"] = False
    sayso.save_training(training)
    with pytest.raises(SystemExit, match="completed full training run"):
        sayso.cmd_select(args())


def test_heldout_runner_reads_the_summary_it_writes(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        out = Path(command[command.index("--out") + 1])
        local = sayso.EVAL_RESULTS / out.relative_to(Path(sayso.HOST) / "lfm/runs/sayso-promotion")
        local.write_text(json.dumps({"passed": 240, "total": 300, "pass_rate": 0.8}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sayso.subprocess, "run", fake_run)
    result = sayso.eval_heldout(training, "/workspace/host/lfm/runs/sayso-promotion/T-full/checkpoint-1400",
                                "checkpoint-1400")
    assert result == {"passed": 240, "total": 300, "pass_rate": 0.8,
                      "result_dir": str(sayso.EVAL_RESULTS / "T-heldout-checkpoint-1400")}
    command = commands[0]
    assert command[command.index("--limit") + 1] == "300"
    assert "/workspace/host/datasets/A.val.jsonl" in command
    assert "CUDA_VISIBLE_DEVICES=" in command and "HIP_VISIBLE_DEVICES=" in command


def test_heldout_without_a_summary_is_a_failure(sayso, monkeypatch):
    training = trained_run(sayso, ["checkpoint-1400"])
    monkeypatch.setattr(sayso.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))
    with pytest.raises(SystemExit, match="produced no summary"):
        sayso.eval_heldout(training, "/x/checkpoint-1400", "checkpoint-1400")


def test_canary_scores_only_smoke_and_never_dev(sayso, tmp_path, monkeypatch):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("x")
    record = {"run_id": "r1", "validated": True, "validated_sha256": sayso.digest(dataset),
              "dataset_path": str(dataset), "rendered_path": str(dataset)}
    monkeypatch.setattr(sayso, "current", lambda: record)
    monkeypatch.setattr(sayso, "write_current", lambda rec: None)
    suites = []
    monkeypatch.setattr(sayso, "eval_checkpoint",
                        lambda checkpoint, out_name, suite, profile=None: (suites.append(suite) or (True, {})))
    sayso.cmd_canary(argparse.Namespace(reuse_checkpoint=False, model="350m"))
    assert suites == ["smoke"]
