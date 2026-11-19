from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ID = "T1"
OUTPUT = f"/workspace/host/lfm/runs/sayso-promotion/{TRAIN_ID}-full"


@pytest.fixture()
def gpu_calls():
    return []


@pytest.fixture()
def sayso(tmp_path, monkeypatch, gpu_calls):
    loader = importlib.machinery.SourceFileLoader("sayso_run", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setattr(module, "TRAINS", tmp_path / "state" / "train")
    monkeypatch.setattr(module, "EVAL_RESULTS", tmp_path / "runs")
    monkeypatch.setattr(module, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(module, "new_id", lambda: "R1")
    monkeypatch.setattr(module, "gpu", gpu_calls.append)
    return module


def run_args(**values):
    return argparse.Namespace(**{"model": "350m", "hypothesis": "shelf lights", "config": "cfg.yaml",
                                 "accept_drift": None, "resume": False, **values})


def choose_checkpoint(sayso, training):
    training.update(selected_checkpoint=f"{training['output']}/checkpoint-1400",
                    auto_eval={"checkpoint-1400": {"promotion_pass_rate": 0.8}})
    sayso.save_training(training)


def install_stages(sayso, monkeypatch, *, eval_passes=True, fail_at=None, stub_select=True):
    calls = []

    def stage(name, effect):
        def fake(args):
            calls.append((name, args))
            if name == fail_at:
                raise SystemExit(f"{name} broke")
            effect(args)
        return fake

    def promote_effect(args):
        sayso.save(sayso.STATE / "promoted_dataset.json", {"run_id": "A"})

    def train_effect(args):
        sayso.save_training({"train_id": TRAIN_ID, "dataset_run_id": "A", "git_commit": "c0de", "profile": "350m",
                             "output": OUTPUT, "trained": True, "manifest": {"max_steps": 2500}})

    def select_effect(args):
        choose_checkpoint(sayso, sayso.training_run(args))

    def eval_effect(args):
        training = sayso.training_run(args)
        comparison = {"passed": eval_passes, "suites": {
            suite: {"total": 10, "candidate_passed": 8, "champion_passed": 6, "wins": [], "losses": []}
            for suite in ("promotion", "context", "regression")}}
        training.update(release={"checkpoint": training["selected_checkpoint"], "gguf": "/x/T1-Q4_K_M.gguf",
                                 "gguf_sha256": "abc123", "comparison": comparison,
                                 "evaluated_at": sayso.utc_now().isoformat()},
                        final_eval_passed=eval_passes)
        sayso.save_training(training)
        if not eval_passes:
            raise SystemExit("Release eval failed; model was not promoted")

    stages = {
        "cmd_generate": ("generate", lambda args: None),
        "cmd_validate": ("validate", lambda args: None),
        "cmd_canary": ("canary", lambda args: None),
        "cmd_promote_dataset": ("promote-dataset", promote_effect),
        "cmd_train": ("train", train_effect),
        "cmd_eval": ("eval", eval_effect),
    }
    if stub_select:
        stages["cmd_select"] = ("select", select_effect)
    for attr, (name, effect) in stages.items():
        monkeypatch.setattr(sayso, attr, stage(name, effect))
    monkeypatch.setattr(sayso, "cmd_promote_model", lambda args: calls.append(("promote-model", args)))
    return calls


def read_status(sayso, capsys):
    capsys.readouterr()
    sayso.cmd_status(argparse.Namespace(json=True))
    return json.loads(capsys.readouterr().out)


IDLE_STATUS = {"run": {}, "training_progress": None, "queued_promotions": [], "next_command": None}


def test_run_chains_every_stage_in_order_and_queues_a_passing_model(sayso, monkeypatch):
    calls = install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args())
    assert [name for name, _ in calls] == [
        "generate", "validate", "canary", "promote-dataset", "train", "select", "eval"]
    seen = dict(calls)
    assert seen["generate"].config == "cfg.yaml" and seen["generate"].accept_drift is None
    assert seen["canary"].model == "350m" and seen["canary"].reuse_checkpoint is False
    assert seen["train"].model == "350m" and seen["train"].chained is True
    assert seen["select"].train_id == TRAIN_ID and seen["eval"].train_id == TRAIN_ID
    run = sayso.load_run()
    assert run["stage"] == "finished" and run["verdict"] == "queued" and run["failure"] is None
    assert run["stages_done"] == ["generate", "validate", "canary", "promote-dataset", "train", "select", "eval",
                                  "queue"]
    assert run["train_id"] == TRAIN_ID and run["dataset_run_id"] == "A"


def test_run_selects_the_checkpoint_by_auto_eval_right_after_training(sayso, monkeypatch):
    calls = install_stages(sayso, monkeypatch, stub_select=False)
    monkeypatch.setattr(sayso, "ensure_vm", lambda: None)

    def auto_eval(training):
        calls.append(("auto_eval", None))
        choose_checkpoint(sayso, training)

    monkeypatch.setattr(sayso, "auto_eval", auto_eval)
    sayso.cmd_run(run_args())
    names = [name for name, _ in calls]
    assert names[names.index("train") + 1] == "auto_eval"
    assert names.index("eval") > names.index("auto_eval")
    assert sayso.load_run()["verdict"] == "queued"


def test_resume_skips_every_stage_already_done(sayso, monkeypatch):
    install_stages(sayso, monkeypatch, fail_at="select")
    with pytest.raises(SystemExit, match="select broke"):
        sayso.cmd_run(run_args())
    assert sayso.load_run()["stages_done"] == ["generate", "validate", "canary", "promote-dataset", "train"]
    calls = install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args(resume=True, hypothesis=None))
    assert [name for name, _ in calls] == ["select", "eval"]
    assert sayso.load_run()["verdict"] == "queued"


def test_a_failing_stage_is_recorded_serves_gpu_and_reraises(sayso, monkeypatch, gpu_calls):
    install_stages(sayso, monkeypatch, fail_at="validate")
    with pytest.raises(SystemExit, match="validate broke"):
        sayso.cmd_run(run_args())
    run = sayso.load_run()
    assert run["stage"] == "validate"
    assert run["failure"] == {"stage": "validate", "reason": "validate broke"}
    assert run["stages_done"] == ["generate"]
    assert gpu_calls == ["serve"]


def test_gpu_serve_errors_do_not_mask_the_stage_error(sayso, monkeypatch):
    def broken_gpu(action):
        raise OSError("lock missing")

    monkeypatch.setattr(sayso, "gpu", broken_gpu)
    install_stages(sayso, monkeypatch, fail_at="generate")
    with pytest.raises(SystemExit, match="generate broke"):
        sayso.cmd_run(run_args())
    assert sayso.load_run()["failure"] == {"stage": "generate", "reason": "generate broke"}


def test_starting_over_an_unfinished_run_refuses_and_points_to_resume(sayso, monkeypatch):
    install_stages(sayso, monkeypatch, fail_at="validate")
    with pytest.raises(SystemExit, match="validate broke"):
        sayso.cmd_run(run_args())
    install_stages(sayso, monkeypatch)
    with pytest.raises(SystemExit, match="--resume"):
        sayso.cmd_run(run_args(hypothesis="another idea"))


def test_resume_refuses_when_there_is_no_run(sayso, monkeypatch):
    install_stages(sayso, monkeypatch)
    with pytest.raises(SystemExit, match="No run to resume"):
        sayso.cmd_run(run_args(resume=True))


def test_resume_refuses_a_finished_run(sayso, monkeypatch):
    install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args())
    with pytest.raises(SystemExit, match="already finished"):
        sayso.cmd_run(run_args(resume=True))


@pytest.mark.parametrize("hypothesis", [None, "   "])
def test_a_new_run_needs_a_non_empty_hypothesis(sayso, monkeypatch, hypothesis):
    calls = install_stages(sayso, monkeypatch)
    with pytest.raises(SystemExit, match="--hypothesis"):
        sayso.cmd_run(run_args(hypothesis=hypothesis))
    assert calls == [] and not (sayso.STATE / "run.json").exists()


def test_a_passing_model_is_queued_and_never_promoted(sayso, monkeypatch):
    calls = install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args())
    assert "promote-model" not in [name for name, _ in calls]
    queued = sayso.load(sayso.STATE / "promotion_queue" / f"{TRAIN_ID}.json")
    assert queued["train_id"] == TRAIN_ID and queued["dataset_run_id"] == "A" and queued["profile"] == "350m"
    assert queued["hypothesis"] == "shelf lights" and queued["checkpoint"] == f"{OUTPUT}/checkpoint-1400"
    assert queued["gguf"] == "/x/T1-Q4_K_M.gguf" and queued["gguf_sha256"] == "abc123"
    assert queued["comparison"]["passed"] is True
    assert set(queued["comparison"]["suites"]) == {"promotion", "context", "regression"}


def test_failed_release_gates_finish_not_promotable_without_a_queue_entry(sayso, monkeypatch, gpu_calls):
    install_stages(sayso, monkeypatch, eval_passes=False)
    sayso.cmd_run(run_args())
    run = sayso.load_run()
    assert run["stage"] == "finished" and run["verdict"] == "not_promotable" and run["failure"] is None
    assert not (sayso.STATE / "promotion_queue").exists()
    assert gpu_calls == []


def test_the_ledger_gets_one_line_per_finished_run_with_its_verdict(sayso, monkeypatch):
    install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args())
    lines = [json.loads(line) for line in sayso.LEDGER.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["run_key"] == "R1" and entry["verdict"] == "queued" and entry["failure"] is None
    assert entry["model"] == "350m" and entry["hypothesis"] == "shelf lights" and entry["config"] == "cfg.yaml"
    assert entry["dataset_run_id"] == "A" and entry["train_id"] == TRAIN_ID and entry["git_commit"] == "c0de"
    assert entry["auto_eval"] == {"checkpoint-1400": {"promotion_pass_rate": 0.8}}
    assert entry["selected_checkpoint"] == f"{OUTPUT}/checkpoint-1400"
    assert entry["release_comparison"]["promotion"] == {"candidate_passed": 8, "champion_passed": 6, "total": 10}
    assert entry["finished_at"]


def test_the_ledger_is_append_only_across_a_failure_and_a_resume(sayso, monkeypatch):
    install_stages(sayso, monkeypatch, fail_at="train")
    with pytest.raises(SystemExit, match="train broke"):
        sayso.cmd_run(run_args())
    first = sayso.LEDGER.read_text(encoding="utf-8")
    install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args(resume=True))
    text = sayso.LEDGER.read_text(encoding="utf-8")
    assert text.startswith(first)
    failed, finished = (json.loads(line) for line in text.splitlines())
    assert failed["verdict"] == "failed" and failed["failure"] == {"stage": "train", "reason": "train broke"}
    assert finished["verdict"] == "queued" and finished["failure"] is None


def test_status_reports_an_idle_pipeline_when_nothing_has_run(sayso, capsys):
    assert read_status(sayso, capsys) == IDLE_STATUS


def test_status_without_json_prints_the_same_object_indented(sayso, capsys):
    sayso.cmd_status(argparse.Namespace(json=False))
    text = capsys.readouterr().out
    assert json.loads(text) == IDLE_STATUS
    assert text.count("\n") > 1


def test_status_points_a_failed_run_at_resume(sayso, monkeypatch, capsys):
    install_stages(sayso, monkeypatch, fail_at="select")
    with pytest.raises(SystemExit, match="select broke"):
        sayso.cmd_run(run_args())
    status = read_status(sayso, capsys)
    assert status["run"]["failure"] == {"stage": "select", "reason": "select broke"}
    assert status["next_command"] == "./sayso run --resume"
    assert status["queued_promotions"] == []


def test_status_points_a_queued_run_at_promote_model(sayso, monkeypatch, capsys):
    install_stages(sayso, monkeypatch)
    sayso.cmd_run(run_args())
    status = read_status(sayso, capsys)
    assert status["run"]["verdict"] == "queued"
    assert status["queued_promotions"] == [TRAIN_ID]
    assert status["next_command"] == f"./sayso promote-model --train-id {TRAIN_ID}"


def test_status_suggests_nothing_after_a_not_promotable_run(sayso, monkeypatch, capsys):
    install_stages(sayso, monkeypatch, eval_passes=False)
    sayso.cmd_run(run_args())
    status = read_status(sayso, capsys)
    assert status["run"]["verdict"] == "not_promotable"
    assert status["queued_promotions"] == []
    assert status["next_command"] is None


def test_status_reports_training_progress_from_checkpoints_and_manifest(sayso, monkeypatch, capsys):
    install_stages(sayso, monkeypatch, fail_at="select")
    with pytest.raises(SystemExit, match="select broke"):
        sayso.cmd_run(run_args())
    directory = sayso.EVAL_RESULTS / f"{TRAIN_ID}-full"
    for name in ("checkpoint-200", "checkpoint-1400", "final"):
        (directory / name).mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"max_steps": 2400}), encoding="utf-8")
    status = read_status(sayso, capsys)
    assert status["training_progress"] == {"train_id": TRAIN_ID, "step": 1400, "max_steps": 2400}


@pytest.mark.parametrize(("chained", "suggests_select"), [(False, True), (True, False)])
def test_train_suggests_select_only_when_run_by_hand(sayso, tmp_path, monkeypatch, capsys, chained, suggests_select):
    for name in ("check_trainable", "ensure_vm", "check_trainer_python", "settle_gpu"):
        monkeypatch.setattr(sayso, name, lambda *args: None)
    monkeypatch.setattr(sayso, "ensure_pinned", lambda: {"git_commit": "c0de", "sayso_commit": "5a75"})
    monkeypatch.setattr(sayso, "render_and_stage", lambda record: "/staged.jsonl")
    monkeypatch.setattr(sayso, "run", lambda command, **kwargs: None)
    dataset = tmp_path / "A.jsonl"
    dataset.write_text("A")
    digest = sayso.digest(dataset)
    sayso.save(sayso.STATE / "A.json", {"run_id": "A", "dataset_path": str(dataset), "dataset_sha256": digest,
                                        "rendered_path": str(dataset), "git_commit": "gen", "config": "cfg.yaml",
                                        "trained": False})
    sayso.save(sayso.STATE / "current.json", {"run_id": "A"})
    sayso.save(sayso.STATE / "promoted_dataset.json",
               {"promoted": True, "validation_passed": True, "canary_passed": True, "run_id": "A",
                "dataset_sha256": digest, "rendered_sha256": digest})
    sayso.cmd_train(argparse.Namespace(chained=chained))
    out = capsys.readouterr().out
    assert "Trained: " in out
    assert ("Next: ./sayso select" in out) is suggests_select
