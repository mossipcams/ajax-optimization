from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str, **kwargs) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, **kwargs).strip()


def snapshot_worktree(root: Path) -> dict:
    repository = Path(_git(root, "rev-parse", "--show-toplevel"))
    prefix = root.resolve().relative_to(repository.resolve())
    with tempfile.TemporaryDirectory(prefix="sayso-index-") as directory:
        index = Path(directory) / "index"
        original = Path(_git(repository, "rev-parse", "--git-path", "index"))
        if not original.is_absolute():
            original = repository / original
        if original.is_file():
            shutil.copyfile(original, index)
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        _git(repository, "-c", "core.fsmonitor=false", "add", "-A", env=env)
        tree = _git(repository, "write-tree", env=env)
    vendor = (prefix / "vendor/SaySo").as_posix()
    entry = _git(repository, "ls-tree", tree, "--", vendor).split()
    if len(entry) < 3 or entry[:2] != ["160000", "commit"]:
        raise SystemExit("Generation needs vendor/SaySo initialized as a Git submodule")
    ref = f"refs/sayso/snapshots/{tree}"
    existing = subprocess.run(["git", "rev-parse", "--verify", ref], cwd=repository,
                              capture_output=True, text=True)
    if existing.returncode == 0:
        commit = existing.stdout.strip()
    else:
        commit = _git(repository, "-c", "user.name=SaySo", "-c", "user.email=sayso@localhost",
                      "commit-tree", tree, input="SaySo generation snapshot\n")
        _git(repository, "update-ref", ref, commit)
    return {"snapshot_commit": commit, "snapshot_tree": tree, "snapshot_ref": ref,
            "sayso_commit": entry[2],
            "generator_tree": _git(repository, "rev-parse", f"{tree}:{prefix / 'training/generators'}")}


@contextmanager
def materialize_snapshot(root: Path, snapshot: dict):
    repository = Path(_git(root, "rev-parse", "--show-toplevel"))
    prefix = root.resolve().relative_to(repository.resolve())
    with tempfile.TemporaryDirectory(prefix="sayso-snapshot-") as directory:
        checkout = Path(directory) / "checkout"
        _git(repository, "-c", "core.hooksPath=/dev/null", "worktree", "add", "--detach",
             str(checkout), snapshot["snapshot_commit"])
        try:
            training_root = checkout / prefix
            vendor = training_root / "vendor/SaySo"
            vendor.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile() as archive:
                subprocess.run(["git", "archive", snapshot["sayso_commit"]],
                               cwd=root / "vendor/SaySo", stdout=archive, check=True)
                archive.seek(0)
                subprocess.run(["tar", "-xf", "-", "-C", str(vendor)], stdin=archive, check=True)
            yield training_root
        finally:
            _git(repository, "worktree", "remove", "--force", str(checkout))


def snapshot_environment(root: Path) -> dict:
    paths = (root, root / "training", root / "vendor/SaySo", root / "vendor/SaySo/satellite")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_") and key not in {"PYTEST_ADDOPTS", "PYTEST_PLUGINS"}}
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONDONTWRITEBYTECODE="1")
    return env


def fingerprint_snapshot(root: Path, python: Path, snapshot: dict, recipe: str) -> dict:
    return json.loads(subprocess.check_output(
        [str(python), "-m", "dataset_gates", recipe, snapshot["generator_tree"], snapshot["sayso_commit"]],
        cwd=root, env=snapshot_environment(root), text=True))


def check_trainable(root: Path, python: Path, record: dict) -> None:
    if record.get("reproduced") is not True:
        raise SystemExit("Full training blocked: candidate was not reproduced; run ./sayso promote-dataset")
    try:
        dataset_sha256 = digest(Path(record["dataset_path"]))
    except (KeyError, OSError) as exc:
        raise SystemExit(f"Full training blocked: cannot read candidate dataset: {exc}") from exc
    if record.get("reproduced_sha256") != dataset_sha256:
        raise SystemExit("Full training blocked: reproduced_sha256 does not match the current dataset")
    required = ("snapshot_commit", "snapshot_tree", "generator_tree", "sayso_commit", "config")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise SystemExit(f"Full training blocked: candidate is missing snapshot metadata: {', '.join(missing)}")
    expected = record.get("input_fingerprint")
    if not isinstance(expected, dict) or not expected.get("sha256"):
        raise SystemExit("Full training blocked: candidate is missing input_fingerprint.sha256")
    try:
        with materialize_snapshot(root, record) as checkout:
            actual = fingerprint_snapshot(checkout, python, record, record["config"])
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Full training blocked: cannot recompute snapshot input fingerprint: {exc}") from exc
    if actual.get("sha256") != expected["sha256"]:
        changed = sorted(key for key in expected.keys() | actual.keys()
                         if key != "sha256" and expected.get(key) != actual.get(key))
        raise SystemExit("Full training blocked: input fingerprint mismatch; differing top-level keys: "
                         + ", ".join(changed or ["sha256"]))
    try:
        repository = Path(_git(root, "rev-parse", "--show-toplevel"))
        prefix = root.resolve().relative_to(repository.resolve())
        generator_tree = _git(repository, "rev-parse", f"HEAD:{prefix / 'training/generators'}")
        vendor = _git(repository, "ls-tree", "HEAD", "--", (prefix / "vendor/SaySo").as_posix()).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Full training blocked: cannot inspect committed generator code: {exc}") from exc
    if generator_tree != record["generator_tree"]:
        raise SystemExit("Full training blocked: committed HEAD generator_tree does not match the candidate snapshot")
    if len(vendor) < 3 or vendor[:2] != ["160000", "commit"] or vendor[2] != record["sayso_commit"]:
        raise SystemExit("Full training blocked: committed HEAD vendor/SaySo gitlink does not match the candidate sayso_commit")


def check_snapshot(root: Path, python: Path, snapshot: dict, state: Path, reason: str | None) -> dict:
    if reason is not None and not reason.strip():
        raise SystemExit("--accept-drift requires a non-empty reason")
    cache = state / "test-cache" / f"{snapshot['snapshot_tree']}.json"
    fingerprint = fingerprint_snapshot(root, python, snapshot, "training/configs/generation/smoke.yaml")
    results = load(cache) if cache.is_file() else {}
    hit = results.get("input_fingerprint") == fingerprint
    print(f"Snapshot tests: cache {'hit' if hit else 'miss'} ({snapshot['snapshot_tree']})", flush=True)
    if not hit:
        results = {"input_fingerprint": fingerprint}
        for name, selection in (("fast", ["training/tests", "-m", "not slow"]),
                                ("golden", ["training/tests/test_generator_golden.py"])):
            report = root / f"{name}-junit.xml"
            command = [str(python), "-m", "pytest", *selection, "-q", "-p", "no:cacheprovider",
                       "--color=no", "--tb=short", f"--junitxml={report}"]
            print("+", " ".join(command), flush=True)
            result = subprocess.run(command, cwd=root, env=snapshot_environment(root),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout, flush=True)
            failures = ET.parse(report).findall(".//failure") if report.is_file() else []
            errors = ET.parse(report).findall(".//error") if report.is_file() else []
            results[name] = {"returncode": result.returncode, "summary": result.stdout[-20000:],
                             "drift": name == "golden" and result.returncode == 1 and bool(failures)
                             and not errors and all("assert " in failure.get("message", "")
                                                    for failure in failures)}
            if name == "fast" and result.returncode:
                save(cache, results)
                raise SystemExit("Fast tests failed; generation stopped (no override)")
        results["input_fingerprint"] = fingerprint_snapshot(
            root, python, snapshot, "training/configs/generation/smoke.yaml")
        save(cache, results)
    if results["fast"]["returncode"]:
        raise SystemExit("Fast tests failed; generation stopped (no override)")
    golden = results["golden"]
    if golden["returncode"]:
        if not golden["drift"]:
            raise SystemExit(f"Golden test could not complete; generation stopped\n{golden['summary']}")
        if reason is None:
            raise SystemExit(f"Golden drift; generation stopped. Use --accept-drift \"reason\" after review.\n"
                             f"{golden['summary']}")
    return {"test_cache": {"hit": hit, "tree": snapshot["snapshot_tree"]},
            "fast_tests_passed": True, "golden_test_passed": golden["returncode"] == 0,
            "accept_drift_reason": reason.strip() if reason is not None else None,
            "golden_diff_summary": golden["summary"] if golden["returncode"] else None}


GATE_PATHS = ("training/tests", "training/generators/validation.py", "training/preflight.py",
              "evals/config/gates.yaml", "training/dataset_gates.py", "sayso")


def check_gate_changes(root: Path, state: Path, snapshot: dict, reason: str | None,
                       run_id: str, candidates: Path) -> dict:
    promoted_path = state / "promoted_dataset.json"
    baseline_run_id = load(promoted_path).get("run_id") if promoted_path.is_file() else None
    baseline_snapshot = load(state / f"{baseline_run_id}.json").get("snapshot_commit") if baseline_run_id else None
    if not baseline_snapshot:
        return {"gate_baseline_run_id": None, "gate_baseline_snapshot": None, "gate_changed_files": [],
                "gate_diff_path": None, "gate_diff_sha256": None}
    repository = Path(_git(root, "rev-parse", "--show-toplevel"))
    prefix = root.resolve().relative_to(repository.resolve())
    paths = [(prefix / path).as_posix() for path in GATE_PATHS]
    refs = [baseline_snapshot, snapshot["snapshot_commit"]]
    flags = ["--no-color", "--no-ext-diff"]
    changed = _git(repository, "diff", *flags, "--name-only", *refs, "--", *paths).splitlines()
    result = {"gate_baseline_run_id": baseline_run_id, "gate_baseline_snapshot": baseline_snapshot,
              "gate_changed_files": changed, "gate_diff_path": None, "gate_diff_sha256": None}
    if not changed:
        return result
    if reason is None:
        raise SystemExit(f"Protected gate files changed since promoted dataset {baseline_run_id}; generation stopped. "
                         "Use --accept-drift \"reason\" after review:\n  " + "\n  ".join(changed))
    patch = _git(repository, "diff", *flags, *refs, "--", *paths) + "\n"
    candidates.mkdir(parents=True, exist_ok=True)
    diff_path = candidates / f"{run_id}.gate-diff.patch"
    diff_path.write_text(patch, encoding="utf-8")
    result.update(gate_diff_path=str(diff_path), gate_diff_sha256=digest(diff_path))
    return result


def _hash_value(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_fingerprint(files: dict[str, Path]) -> dict:
    hashes = {name: digest(path) for name, path in sorted(files.items())}
    return {"sha256": _hash_value(hashes), "files": hashes}


def input_fingerprint(root: Path, recipe: str, generator_tree: str, sayso_commit: str) -> dict:
    from evals import cases
    from generators.config import GeneratorConfig

    config = GeneratorConfig.from_yaml(root / recipe, repo_root=root)
    packages = sorted({f"{item.metadata['Name']}=={item.version}"
                       for item in importlib.metadata.distributions()})
    versions = {}
    for name in ("tokenizers", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    case_files = {case.source: cases.CASES_DIR / case.source for case in cases.load_all_cases()}
    tokenizer_root = root / "training" / config.tokenizer_model
    if not tokenizer_root.is_dir():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(config.tokenizer_model, "tokenizer_config.json")
            tokenizer_root = Path(cached).parent if isinstance(cached, str) else None
        except ImportError:
            tokenizer_root = None
    tokenizer_files = {}
    if tokenizer_root is not None:
        for path in tokenizer_root.rglob("*"):
            if path.is_file() and ("token" in path.name or "vocab" in path.name
                                   or path.name in {"config.json", "merges.txt"}
                                   or path.suffix in {".model", ".jinja", ".py"}):
                tokenizer_files[path.relative_to(tokenizer_root).as_posix()] = path
    inputs = {"recipe": root / recipe, "schema": config.schema_path}
    for name in ("area_distribution_path", "real_home_path"):
        path = getattr(config, name)
        if path is not None:
            inputs[name] = path
    fingerprint = {"generator_tree": generator_tree, "recipe_sha256": config.recipe_hash(),
                   "config_files": _file_fingerprint(inputs),
                   "eval_cases": {"sayso_commit": sayso_commit, **_file_fingerprint(case_files)},
                   "python_version": platform.python_version(),
                   "packages": {"sha256": _hash_value(packages), "installed": packages},
                   "tokenizer": {"model": config.tokenizer_model, **versions,
                                 **_file_fingerprint(tokenizer_files)}}
    return {**fingerprint, "sha256": _hash_value(fingerprint)}


if __name__ == "__main__":
    print(json.dumps(input_fingerprint(Path(__file__).resolve().parents[1], *sys.argv[1:])))
