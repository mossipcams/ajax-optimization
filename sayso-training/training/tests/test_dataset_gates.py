from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

import dataset_gates

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def sayso(tmp_path, monkeypatch):
    loader = importlib.machinery.SourceFileLoader("sayso_gates", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setattr(module, "CANDIDATES", tmp_path / "candidates")
    monkeypatch.setattr(module, "PYTHON", python)
    return module


def git(*command, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           *command], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def make_repo(tmp_path):
    repository = tmp_path / "repo"
    root = repository / "sayso-training"
    vendor = root / "vendor" / "SaySo"
    repository.mkdir()
    git("init", "-q", "-b", "main", cwd=repository)
    vendor.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=vendor)
    (vendor / "README").write_text("sayso\n", encoding="utf-8")
    git("add", "README", cwd=vendor)
    git("commit", "-qm", "sayso", cwd=vendor)
    (root / "training" / "generators").mkdir(parents=True)
    (root / "training" / "generators" / "row.py").write_text("ROWS = 1\n", encoding="utf-8")
    (root / "training" / "dataset_gates.py").write_text("GATE = 1\n", encoding="utf-8")
    (root / "training" / "tracked.txt").write_text("v1\n", encoding="utf-8")
    commit_all(repository, "init")
    return root


def commit_all(repository, message):
    git("add", "-A", cwd=repository)
    git("commit", "-qm", message, cwd=repository)


def stub_pytest(monkeypatch, *, fast=0, golden=0):
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        if command[1:3] != ["-m", "pytest"]:
            return real_run(command, *args, **kwargs)
        is_golden = "training/tests/test_generator_golden.py" in command
        returncode = golden if is_golden else fast
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        failure = '<failure message="assert 1 == 2">drift</failure>' if is_golden and returncode else ""
        report.write_text(f'<testsuite><testcase name="t">{failure}</testcase></testsuite>', encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout=f"pytest rc={returncode}\n")

    monkeypatch.setattr(dataset_gates.subprocess, "run", run)


def stub_fingerprint(monkeypatch, sayso, sha="fp"):
    def fingerprint(root, python, snapshot, recipe):
        return {"sha256": sha, "recipe_sha256": "recipe"}

    monkeypatch.setattr(sayso, "fingerprint_snapshot", fingerprint)
    monkeypatch.setattr(dataset_gates, "fingerprint_snapshot", fingerprint)


def stub_generate(monkeypatch, sayso, corpus=b"candidate\n", dataset_hash="h1"):
    calls = []

    def generate(checkout, config, dataset, manifest):
        calls.append(config)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_bytes(corpus)
        manifest.write_text(json.dumps({"dataset_hash": dataset_hash}), encoding="utf-8")

    monkeypatch.setattr(sayso, "generate_dataset", generate)
    return calls


def candidate(sayso, snapshot):
    dataset = sayso.CANDIDATES / "A.jsonl"
    manifest = sayso.CANDIDATES / "A.manifest.json"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_bytes(b"candidate\n")
    manifest.write_text(json.dumps({"dataset_hash": "h1"}), encoding="utf-8")
    digest = dataset_gates.digest(dataset)
    record = {"run_id": "A", "dataset_path": str(dataset), "manifest_path": str(manifest),
              "rendered_path": str(dataset), "rendered_sha256": digest, "staged_data": "/staged.jsonl",
              "git_commit": "gen", "config": "cfg.yaml", "canary_metrics": {}, "dataset_sha256": digest,
              "validated": True, "validated_sha256": digest, "canary_passed": True, **snapshot}
    sayso.write_current(record)
    return record


def trainable(tmp_path, snapshot):
    dataset = tmp_path / "A.jsonl"
    dataset.write_bytes(b"candidate\n")
    return {"run_id": "A", "dataset_path": str(dataset), "reproduced": True,
            "reproduced_sha256": dataset_gates.digest(dataset), "input_fingerprint": {"sha256": "fp"},
            "config": "cfg.yaml", **snapshot}


def test_generate_refuses_golden_drift_without_accept_drift(sayso, tmp_path, monkeypatch):
    monkeypatch.setattr(sayso, "ROOT", make_repo(tmp_path))
    stub_pytest(monkeypatch, golden=1)
    stub_fingerprint(monkeypatch, sayso)
    generated = stub_generate(monkeypatch, sayso)
    with pytest.raises(SystemExit, match="Golden drift; generation stopped"):
        sayso.cmd_generate(argparse.Namespace(accept_drift=None))
    assert generated == []
    assert not (sayso.STATE / "current.json").exists()


@pytest.mark.parametrize("reason", ["", "   "])
def test_generate_refuses_a_blank_accept_drift_reason(sayso, tmp_path, monkeypatch, reason):
    monkeypatch.setattr(sayso, "ROOT", make_repo(tmp_path))
    stub_pytest(monkeypatch, golden=1)
    stub_fingerprint(monkeypatch, sayso)
    generated = stub_generate(monkeypatch, sayso)
    with pytest.raises(SystemExit, match="--accept-drift requires a non-empty reason"):
        sayso.cmd_generate(argparse.Namespace(accept_drift=reason))
    assert generated == []
    assert not (sayso.STATE / "current.json").exists()


def test_generate_with_a_reason_accepts_golden_drift_and_records_it(sayso, tmp_path, monkeypatch):
    monkeypatch.setattr(sayso, "ROOT", make_repo(tmp_path))
    stub_pytest(monkeypatch, golden=1)
    stub_fingerprint(monkeypatch, sayso)
    generated = stub_generate(monkeypatch, sayso)
    sayso.cmd_generate(argparse.Namespace(accept_drift="  renamed golden row  "))
    record = sayso.current()
    assert record["accept_drift_reason"] == "renamed golden row"
    assert record["golden_test_passed"] is False
    assert record["golden_diff_summary"]
    assert generated == [sayso.GENERATION_CONFIG]


def test_fast_test_failure_refuses_even_with_accept_drift(sayso, tmp_path, monkeypatch):
    monkeypatch.setattr(sayso, "ROOT", make_repo(tmp_path))
    stub_pytest(monkeypatch, fast=1, golden=1)
    stub_fingerprint(monkeypatch, sayso)
    generated = stub_generate(monkeypatch, sayso)
    with pytest.raises(SystemExit, match=re.escape("Fast tests failed; generation stopped (no override)")):
        sayso.cmd_generate(argparse.Namespace(accept_drift="renamed golden row"))
    assert generated == []
    assert not (sayso.STATE / "current.json").exists()


def test_gate_change_refuses_without_reason_and_saves_the_diff_with_one(sayso, tmp_path):
    root = make_repo(tmp_path)
    baseline = dataset_gates.snapshot_worktree(root)
    sayso.save(sayso.STATE / "A.json", {"run_id": "A", **baseline})
    sayso.save(sayso.STATE / "promoted_dataset.json", {"promoted": True, "run_id": "A"})
    (root / "training" / "dataset_gates.py").write_text("GATE = 2\n", encoding="utf-8")
    snapshot = dataset_gates.snapshot_worktree(root)
    candidates = tmp_path / "candidates"
    with pytest.raises(SystemExit, match="Protected gate files changed since promoted dataset A"):
        dataset_gates.check_gate_changes(root, sayso.STATE, snapshot, None, "B", candidates)
    assert not candidates.exists()
    result = dataset_gates.check_gate_changes(root, sayso.STATE, snapshot, "tightened gate", "B", candidates)
    diff = candidates / "B.gate-diff.patch"
    assert result["gate_baseline_run_id"] == "A"
    assert result["gate_changed_files"] == ["sayso-training/training/dataset_gates.py"]
    assert result["gate_diff_path"] == str(diff)
    assert "+GATE = 2" in diff.read_text(encoding="utf-8")
    assert result["gate_diff_sha256"] == dataset_gates.digest(diff)


def test_gate_check_has_no_baseline_before_anything_is_promoted(sayso, tmp_path):
    root = make_repo(tmp_path)
    (root / "training" / "dataset_gates.py").write_text("GATE = 2\n", encoding="utf-8")
    snapshot = dataset_gates.snapshot_worktree(root)
    result = dataset_gates.check_gate_changes(root, sayso.STATE, snapshot, None, "A", tmp_path / "candidates")
    assert result == {"gate_baseline_run_id": None, "gate_baseline_snapshot": None, "gate_changed_files": [],
                      "gate_diff_path": None, "gate_diff_sha256": None}


def test_promotion_refuses_when_the_regenerated_corpus_differs(sayso, tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(sayso, "ROOT", root)
    candidate(sayso, dataset_gates.snapshot_worktree(root))
    stub_generate(monkeypatch, sayso, corpus=b"regenerated\n")
    with pytest.raises(SystemExit, match="Dataset reproduction failed: candidate sha256="):
        sayso.cmd_promote_dataset(argparse.Namespace())
    assert not (sayso.STATE / "promoted_dataset.json").exists()
    assert "reproduced" not in sayso.current()


def test_promotion_refuses_a_candidate_without_a_snapshot(sayso, tmp_path):
    candidate(sayso, {})
    with pytest.raises(SystemExit, match="Candidate has no saved snapshot"):
        sayso.cmd_promote_dataset(argparse.Namespace())
    assert not (sayso.STATE / "promoted_dataset.json").exists()


def test_promotion_records_reproduced_when_the_corpus_matches(sayso, tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(sayso, "ROOT", root)
    record = candidate(sayso, dataset_gates.snapshot_worktree(root))
    stub_generate(monkeypatch, sayso)
    sayso.cmd_promote_dataset(argparse.Namespace())
    promotion = dataset_gates.load(sayso.STATE / "promoted_dataset.json")
    assert promotion["run_id"] == "A" and promotion["dataset_sha256"] == record["dataset_sha256"]
    current = sayso.current()
    assert current["reproduced"] is True
    assert current["reproduced_sha256"] == record["dataset_sha256"]


def test_training_accepts_a_candidate_that_matches_its_snapshot(sayso, tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    record = trainable(tmp_path, dataset_gates.snapshot_worktree(root))
    stub_fingerprint(monkeypatch, sayso)
    dataset_gates.check_trainable(root, sayso.PYTHON, record)


def test_training_refuses_an_unreproduced_candidate(sayso, tmp_path):
    root = make_repo(tmp_path)
    record = trainable(tmp_path, dataset_gates.snapshot_worktree(root))
    record["reproduced"] = False
    with pytest.raises(SystemExit, match="candidate was not reproduced"):
        dataset_gates.check_trainable(root, sayso.PYTHON, record)


def test_training_refuses_a_reproduced_sha256_that_differs_from_the_dataset(sayso, tmp_path):
    root = make_repo(tmp_path)
    record = trainable(tmp_path, dataset_gates.snapshot_worktree(root))
    record["reproduced_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="reproduced_sha256 does not match"):
        dataset_gates.check_trainable(root, sayso.PYTHON, record)


def test_training_refuses_a_fingerprint_mismatch(sayso, tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    record = trainable(tmp_path, dataset_gates.snapshot_worktree(root))
    stub_fingerprint(monkeypatch, sayso, sha="other")
    with pytest.raises(SystemExit, match="input fingerprint mismatch"):
        dataset_gates.check_trainable(root, sayso.PYTHON, record)


def test_training_refuses_when_committed_generators_moved_on(sayso, tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    record = trainable(tmp_path, dataset_gates.snapshot_worktree(root))
    stub_fingerprint(monkeypatch, sayso)
    (root / "training" / "generators" / "row.py").write_text("ROWS = 2\n", encoding="utf-8")
    commit_all(root.parent, "generator change")
    with pytest.raises(SystemExit, match="committed HEAD generator_tree does not match"):
        dataset_gates.check_trainable(root, sayso.PYTHON, record)


def test_train_refuses_a_fingerprint_mismatch_before_launching(sayso, tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(sayso, "ROOT", root)
    record = trainable(tmp_path, dataset_gates.snapshot_worktree(root))
    sayso.write_current(record)
    sayso.save(sayso.STATE / "promoted_dataset.json",
               {"promoted": True, "validation_passed": True, "canary_passed": True, "run_id": "A",
                "dataset_sha256": record["reproduced_sha256"]})
    stub_fingerprint(monkeypatch, sayso, sha="other")
    monkeypatch.setattr(sayso, "ensure_vm", lambda *args: None)
    monkeypatch.setattr(sayso, "check_trainer_python", lambda *args: None)
    monkeypatch.setattr(sayso, "run", lambda *args, **kwargs: pytest.fail("training launched"))
    with pytest.raises(SystemExit, match="input fingerprint mismatch"):
        sayso.cmd_train(argparse.Namespace())


def test_snapshot_leaves_head_index_and_worktree_unchanged(tmp_path):
    root = make_repo(tmp_path)
    repository = root.parent
    (root / "training" / "tracked.txt").write_text("v2\n", encoding="utf-8")
    (root / "training" / "new.txt").write_text("new\n", encoding="utf-8")
    (root / "training" / "staged.txt").write_text("staged\n", encoding="utf-8")
    git("add", "training/staged.txt", cwd=root)

    def observe():
        return {"head": git("rev-parse", "HEAD", cwd=repository),
                "index": git("ls-files", "--stage", cwd=repository),
                "staged": git("diff", "--cached", "--name-status", cwd=repository),
                "status": git("status", "--porcelain", cwd=repository),
                "files": {path.name: path.read_text(encoding="utf-8")
                          for path in (root / "training").glob("*.txt")}}

    before = observe()
    snapshot = dataset_gates.snapshot_worktree(root)
    assert observe() == before
    assert snapshot["snapshot_ref"] == f"refs/sayso/snapshots/{snapshot['snapshot_tree']}"
    assert git("rev-parse", "--verify", snapshot["snapshot_ref"], cwd=repository) == snapshot["snapshot_commit"]
    assert git("rev-parse", f"{snapshot['snapshot_commit']}^{{tree}}", cwd=repository) == snapshot["snapshot_tree"]
    for name in ("tracked.txt", "new.txt", "staged.txt"):
        contents = git("show", f"{snapshot['snapshot_tree']}:sayso-training/training/{name}", cwd=repository)
        assert contents == before["files"][name].strip()
