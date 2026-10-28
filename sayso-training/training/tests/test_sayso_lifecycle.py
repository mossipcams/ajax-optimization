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
    loader = importlib.machinery.SourceFileLoader("sayso_lifecycle", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setattr(module, "TRAINS", tmp_path / "state" / "train")
    monkeypatch.setattr(module, "EVAL_RESULTS", tmp_path / "runs")
    for name in ("ensure_vm", "gpu", "settle_gpu"):
        monkeypatch.setattr(module, name, lambda *args: None)
    monkeypatch.setattr(module, "ensure_pinned", lambda: {"git_commit": "c0de", "sayso_commit": "5a75"})
    monkeypatch.setattr(module, "render_and_stage", lambda record: "/staged.jsonl")
    monkeypatch.setattr(module, "run", lambda command, **kwargs: None)
    return module


def args(**values):
    return argparse.Namespace(**{"train_id": None, "manual_override": False, "reason": None, **values})


def promote_dataset(sayso, tmp_path, run_id):
    dataset = tmp_path / f"{run_id}.jsonl"
    dataset.write_text(run_id)
    record = {"run_id": run_id, "dataset_path": str(dataset), "dataset_sha256": sayso.digest(dataset),
              "rendered_path": str(dataset), "git_commit": "gen", "config": "cfg.yaml"}
    sayso.write_current(record)
    sayso.save(sayso.STATE / "promoted_dataset.json",
               {"promoted": True, "validation_passed": True, "canary_passed": True, "run_id": run_id,
                "dataset_sha256": record["dataset_sha256"], "rendered_sha256": record["dataset_sha256"]})
    return record


def test_training_leaves_the_newer_candidate_current(sayso, tmp_path):
    promote_dataset(sayso, tmp_path, "A")
    newer = {"run_id": "B", "dataset_path": str(tmp_path / "A.jsonl")}
    sayso.write_current(newer)
    sayso.cmd_train(args())
    assert sayso.load(sayso.STATE / "current.json") == {"run_id": "B"}
    training = sayso.training_run(args())
    assert training["dataset_run_id"] == "A"
    assert training["trained"] is True
    assert training["git_commit"] == "c0de" and training["sayso_commit"] == "5a75"
    assert sayso.load(sayso.STATE / "A.json")["trained"] is True


def test_eval_targets_the_trained_run_not_a_candidate_generated_afterwards(sayso, tmp_path):
    promote_dataset(sayso, tmp_path, "A")
    sayso.cmd_train(args())
    sayso.write_current({"run_id": "B", "dataset_path": str(tmp_path / "A.jsonl")})
    with pytest.raises(SystemExit, match="selected checkpoint"):
        sayso.cmd_eval(args())
    assert sayso.training_run(args())["dataset_run_id"] == "A"


def test_retraining_cannot_be_promoted_with_the_previous_runs_eval(sayso, tmp_path):
    promote_dataset(sayso, tmp_path, "A")
    sayso.cmd_train(args())
    first = sayso.training_run(args())
    gguf = tmp_path / "first.gguf"
    gguf.write_bytes(b"GGUF")
    first.update(release={"checkpoint": "x", "gguf": str(gguf), "gguf_sha256": sayso.digest(gguf)},
                 final_metrics={"promotion": {}}, final_eval_passed=False)
    sayso.save_training(first)
    sayso.cmd_train(args())
    assert sayso.training_run(args())["train_id"] != first["train_id"]
    with pytest.raises(SystemExit, match="completed release evaluation"):
        sayso.cmd_promote_model(args(manual_override=True, reason="ship it"))


def test_promotion_refuses_a_gguf_that_changed_after_eval(sayso, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    training = {"train_id": "T", "trained": True, "final_metrics": {"promotion": {}}, "final_eval_passed": True,
                "release": {"checkpoint": "x", "gguf": str(gguf), "gguf_sha256": sayso.digest(gguf)}}
    sayso.save_training(training)
    gguf.write_bytes(b"GGUF changed")
    with pytest.raises(SystemExit, match="changed after it was evaluated"):
        sayso.cmd_promote_model(args())


def test_select_prefers_fewer_safety_flags_then_pass_rate_then_earlier_step(sayso):
    table = {"checkpoint-200": {"pass_rate": 0.9, "safety_flags": 1},
             "checkpoint-400": {"pass_rate": 0.7, "safety_flags": 0},
             "checkpoint-600": {"pass_rate": 0.7, "safety_flags": 0},
             "final": {"pass_rate": 0.6, "safety_flags": 0}}
    assert sayso.select_checkpoint(table) == "checkpoint-400"


def test_select_scores_every_checkpoint_on_dev(sayso, tmp_path, monkeypatch):
    training = {"train_id": "T", "trained": True, "output": "/workspace/host/lfm/runs/sayso-promotion/T-full"}
    sayso.save_training(training)
    for name in ("checkpoint-200", "checkpoint-1000", "checkpoint-400"):
        (sayso.local_output(training) / name).mkdir(parents=True)
    rates = {"checkpoint-200": 0.5, "checkpoint-400": 0.8, "checkpoint-1000": 0.8, "final": 0.7}
    seen = []

    def fake_eval(checkpoint, out_name, suite, profile=None):
        name = Path(checkpoint).name
        seen.append((name, suite))
        return True, {"dev": {"summary": {"overall_pass_rate": rates[name], "flags": {}}, "result_dir": name}}

    monkeypatch.setattr(sayso, "eval_checkpoint", fake_eval)
    monkeypatch.setattr(sayso, "safety_flags", lambda: {"action_on_ambiguity"})
    sayso.cmd_select(args())
    assert seen == [(name, "dev") for name in ("checkpoint-200", "checkpoint-400", "checkpoint-1000", "final")]
    assert sayso.training_run(args())["selected_checkpoint"].endswith("/checkpoint-400")


def rows(passed, category="c", flags=None):
    return [{"case_id": f"case-{i}", "category": category, "passed": value, "outcome": "x",
             "flags": (flags or {}).get(i, [])} for i, value in enumerate(passed)]


def compare(sayso, candidate, champion, context_passed=True):
    return sayso.compare_release(candidate, champion, context_passed=context_passed, safety={"unsafe"})


def test_release_needs_more_wins_than_losses_on_promotion(sayso):
    champion = {"promotion": rows([True, False, False, True])}
    better = compare(sayso, {"promotion": rows([True, True, False, True])}, champion)
    assert better["passed"] and better["suites"]["promotion"]["wins"] == ["case-1"]
    tie = compare(sayso, {"promotion": rows([False, True, False, True])}, champion)
    assert not tie["passed"] and "needs more wins than losses" in tie["failures"][0]


def test_release_rejects_new_safety_flags_even_when_better(sayso):
    champion = {"promotion": rows([False, False, True])}
    candidate = {"promotion": rows([True, True, True], flags={2: ["unsafe", "harmless"]})}
    report = compare(sayso, candidate, champion)
    assert report["suites"]["promotion"]["new_safety_flags"] == ["case-2:unsafe"]
    assert not report["passed"]


def test_release_limits_regression_losses_categories_and_context(sayso):
    champion = {"promotion": rows([False] * 6 + [True] * 3, category="a"),
                "regression": rows([True] * 4)}
    candidate = {"promotion": rows([True] * 6 + [False] * 3, category="a"),
                 "regression": rows([False] * 3 + [True])}
    report = compare(sayso, candidate, champion, context_passed=False)
    assert not report["passed"]
    assert any("regression: net loss of 3" in failure for failure in report["failures"])
    assert any("context" in failure for failure in report["failures"])
    assert report["suites"]["promotion"]["category_net"] == {"a": 3}
    assert sayso.sign_test_p(0, 0) == 1.0
    assert sayso.sign_test_p(10, 0) == pytest.approx(2 / 1024)


def git(*command, cwd):
    subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
                        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_ALLOW_PROTOCOL": "file"})


def make_checkout(tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    super_root = tmp_path / "repo"
    sayso_dir = super_root / "training" / "vendor" / "SaySo"
    sayso_dir.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=sayso_dir)
    (sayso_dir / "f").write_text("1")
    git("add", "f", cwd=sayso_dir)
    git("commit", "-qm", "one", cwd=sayso_dir)
    git("push", "-q", str(remote), "main", cwd=sayso_dir)
    git("init", "-q", "-b", "main", cwd=super_root)
    (super_root / ".gitmodules").write_text(
        f'[submodule "training/vendor/SaySo"]\n\tpath = training/vendor/SaySo\n\turl = {remote}\n')
    (super_root / "tracked").write_text("x")
    git("add", ".gitmodules", "tracked", cwd=super_root)
    git("commit", "-qm", "init", cwd=super_root)
    return super_root / "training", sayso_dir


def test_pinning_accepts_a_clean_tree_with_a_pushed_sayso_commit(sayso, tmp_path):
    root, sayso_dir = make_checkout(tmp_path)
    loader = importlib.machinery.SourceFileLoader("sayso_pin", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    fresh = importlib.util.module_from_spec(spec)
    loader.exec_module(fresh)
    pins = fresh.ensure_pinned(root, sayso_dir)
    assert len(pins["sayso_commit"]) == 40 and len(pins["git_commit"]) == 40

    (sayso_dir / "f").write_text("2")
    git("commit", "-qam", "local only", cwd=sayso_dir)
    with pytest.raises(SystemExit, match="push it first"):
        fresh.ensure_pinned(root, sayso_dir)

    (root.parent / "tracked").write_text("edited")
    with pytest.raises(SystemExit, match="tracked changes"):
        fresh.ensure_pinned(root, sayso_dir)


def test_canary_diagnostics_never_touch_locked_suites(sayso, tmp_path, monkeypatch):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("x")
    record = {"run_id": "r1", "validated": True, "validated_sha256": sayso.digest(dataset),
              "dataset_path": str(dataset), "rendered_path": str(dataset)}
    monkeypatch.setattr(sayso, "current", lambda: record)
    monkeypatch.setattr(sayso, "write_current", lambda rec: None)
    suites = []
    monkeypatch.setattr(sayso, "eval_checkpoint",
                        lambda checkpoint, out_name, suite, profile=None: (suites.append(suite) or (True, {})))
    manifest = sayso.EVAL_RESULTS / "r1-canary" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"final_step": 250, "max_steps": 2500, "last_logged_learning_rate": 5e-5}))
    sayso.cmd_canary(argparse.Namespace(reuse_checkpoint=False))
    assert suites == ["smoke", "dev"]
    assert record["canary_schedule"] == {"final_step": 250, "max_steps": 2500, "last_logged_learning_rate": 5e-5}
