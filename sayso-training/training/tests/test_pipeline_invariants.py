from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import dataset_gates as gates

TESTS = Path(__file__).resolve().parent
TRAINING = TESTS.parent
SAYSO_TRAINING = TRAINING.parent
DOCS = SAYSO_TRAINING / "docs" / "PIPELINE.md"
ACTIVE_TRAINER = TRAINING / "scripts" / "sayso_trainer.py"
SPLIT_SCRIPT = TRAINING / "scripts" / "split_dataset.py"
SMOKE_FIXTURE = TRAINING / "fixtures" / "sayso_lfm_smoke.jsonl"
BASELINE = TRAINING / "configs" / "training_baseline.json"
MODELS = TRAINING / "configs" / "models.yaml"
SCHEMA_V2 = SAYSO_TRAINING / "schemas" / "sayso-tool-schema-v2.json"
DATASET_GATES = TRAINING / "dataset_gates.py"
SAYSO_FILE = SAYSO_TRAINING / "sayso"

DOC_TEXT = DOCS.read_text(encoding="utf-8")
_INVARIANTS_MATCH = re.search(r"^## Pipeline invariants\s*$(.*?)(?=^## |\Z)", DOC_TEXT, re.M | re.S)
assert _INVARIANTS_MATCH is not None, "PIPELINE.md has no 'Pipeline invariants' section"
INVARIANTS = _INVARIANTS_MATCH.group(1)

_SAYSO_LOADER = importlib.machinery.SourceFileLoader("sayso_cli", str(SAYSO_TRAINING / "sayso"))
SAYSO = importlib.util.module_from_spec(importlib.util.spec_from_loader("sayso_cli", _SAYSO_LOADER))
_SAYSO_LOADER.exec_module(SAYSO)


def _doc_match(pattern: str, text: str, what: str) -> re.Match[str]:
    match = re.search(pattern, text, re.S)
    assert match is not None, f"PIPELINE.md does not state {what}"
    return match


def _doc_profile_models() -> dict[str, str]:
    section = _doc_match(r"(?m)^1\. \*\*.*?(?=^2\. \*\*)", INVARIANTS, "invariant 1").group(0)
    models = dict(re.findall(r"`([0-9a-z]+)` = `([A-Za-z0-9._\-]+)`", section))
    assert models, "PIPELINE.md invariant 1 names no model profile"
    return models


def _doc_profile_recipes() -> dict[str, tuple[str, ...]]:
    rows = re.findall(
        r"^[ \t]*\| `([0-9a-z]+)`[^|]*\| `([A-Za-z0-9._\-]+)` \| `([0-9.e\-]+)` \| `([0-9.]+)` \| `([0-9]+)` \| `([0-9]+)` \|",
        DOC_TEXT,
        re.M,
    )
    assert rows, "PIPELINE.md has no model profile recipe table"
    return {row[0]: row[1:] for row in rows}


def _yaml_profiles() -> dict[str, dict]:
    import yaml

    return yaml.safe_load(MODELS.read_text(encoding="utf-8"))["models"]


def _doc_val_ratio() -> float:
    return float(_doc_match(r"VAL_RATIO = ([0-9.]+)", INVARIANTS, "VAL_RATIO").group(1))


def _doc_generation_dir() -> str:
    return _doc_match(r"under `(training/configs/generation/)`", INVARIANTS, "the generation config directory").group(1)


def _doc_generation_config() -> str:
    return _doc_match(
        r"`(training/configs/generation/[^`]+)`,?\s+named by the\s+`GENERATION_CONFIG`", DOC_TEXT, "GENERATION_CONFIG"
    ).group(1)


def _doc_release_slack() -> int:
    return int(_doc_match(r"RELEASE_SLACK = (\d+)", DOC_TEXT, "RELEASE_SLACK").group(1))


def _doc_release_quant() -> str:
    return _doc_match(r"RELEASE_QUANT = ([A-Z0-9_]+)", DOC_TEXT, "RELEASE_QUANT").group(1)


def _doc_release_suites() -> tuple[str, ...]:
    sentence = _doc_match(r"on each of (.*?)\(`RELEASE_SUITES`\)", DOC_TEXT, "RELEASE_SUITES").group(1)
    return tuple(re.findall(r"`([a-z_]+)`", sentence))


def _joined_constants(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return ""


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"sayso has no function {name}")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _model_defaults(path: Path) -> list[str]:
    defaults = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--model":
                defaults.extend(_joined_constants(kw.value) for kw in node.keywords if kw.arg == "default")
    return defaults


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def test_canary_and_train_run_the_active_trainer() -> None:
    source = (SAYSO_TRAINING / "sayso").read_text(encoding="utf-8")
    assert "training/scripts/sayso_trainer.py" in source
    assert "train_unsloth_full.py" not in source
    tree = ast.parse(source)
    for name in ("cmd_canary", "cmd_train"):
        runs = [
            call for call in ast.walk(_function(tree, name))
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "run"
        ]
        assert any("training/scripts/sayso_trainer.py" in ast.unparse(call) for call in runs), (
            f"{name} does not run training/scripts/sayso_trainer.py"
        )


def test_base_model_matches_doc_in_sayso_and_trainer() -> None:
    profiles = _yaml_profiles()
    doc_models = _doc_profile_models()
    yaml_models = {name: Path(profile["trainer_model"]).name for name, profile in profiles.items()}
    assert doc_models == yaml_models, f"PIPELINE.md profile base models {doc_models} != models.yaml {yaml_models}"
    defaults = _model_defaults(ACTIVE_TRAINER)
    assert defaults, f"{ACTIVE_TRAINER.name} has no --model default"
    for default in defaults:
        assert default.rsplit("/", 1)[-1] == doc_models[SAYSO.DEFAULT_MODEL], (
            f"--model default {default!r} != doc {doc_models[SAYSO.DEFAULT_MODEL]!r}"
        )


def test_active_trainer_loads_bf16_without_fp32_master_weights() -> None:
    tree = ast.parse(ACTIVE_TRAINER.read_text(encoding="utf-8"))
    imported: set[str] = set()
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        if isinstance(node, ast.Call):
            calls.append((ast.unparse(node.func), {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}))
    loads = [kwargs for name, kwargs in calls if name == "AutoModelForCausalLM.from_pretrained"]
    configs = [kwargs for name, kwargs in calls if name == "SFTConfig"]
    assert any("dtype" in kwargs and ast.unparse(kwargs["dtype"]) == "torch.bfloat16" for kwargs in loads), (
        "AutoModelForCausalLM.from_pretrained(dtype=torch.bfloat16) missing"
    )
    assert not any(_is_true(kwargs.get("float32_mixed_precision")) for _, kwargs in calls), (
        "float32_mixed_precision=True must not be passed"
    )
    assert any(_is_true(kwargs.get("bf16")) for kwargs in configs), "SFTConfig(bf16=True) missing"
    assert "peft" not in imported, "active trainer imports peft"


def test_trainer_has_no_adapter_path() -> None:
    tree = ast.parse(ACTIVE_TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func).lower()
            assert "peft" not in callee and "lora" not in callee, f"adapter call in trainer: {callee}"
            for kw in node.keywords:
                assert kw.arg is None or "lora" not in kw.arg.lower(), f"adapter kwarg in trainer: {kw.arg}"


def test_val_ratio_matches_doc() -> None:
    assert SAYSO.VAL_RATIO == _doc_val_ratio(), f"VAL_RATIO {SAYSO.VAL_RATIO} != doc {_doc_val_ratio()}"
    assert SAYSO.VAL_RATIO == 0.05


def _scenario_row(index: int, variant: int) -> dict:
    return {
        "messages": [{"role": "user", "content": f"turn on lamp {index} please variant {variant}"}],
        "metadata": {
            "family": "direct",
            "home_id": f"home_{index:06d}",
            "semantic_id": f"sem_{index:06d}",
        },
    }


def _split(tmp_path: Path, records: list[dict]) -> dict[str, list[dict]]:
    source = tmp_path / "dataset.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    out_dir = tmp_path / "split"
    result = subprocess.run(
        [
            sys.executable,
            str(SPLIT_SCRIPT),
            str(source),
            "--out-dir",
            str(out_dir),
            "--train-ratio",
            str(1 - SAYSO.VAL_RATIO),
            "--val-ratio",
            str(SAYSO.VAL_RATIO),
        ],
        cwd=str(SAYSO_TRAINING),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return {name: _load_jsonl(out_dir / f"sayso_{name}.jsonl") for name in ("train", "val", "test")}


def test_split_keeps_scenario_groups_out_of_both_splits(tmp_path: Path) -> None:
    records = [_scenario_row(index, variant) for index in range(400) for variant in range(3)]
    splits = _split(tmp_path, records)
    assert sum(len(rows) for rows in splits.values()) == len(records)
    assert splits["train"] and splits["val"]
    where: dict[str, set[str]] = {}
    for name, rows in splits.items():
        for row in rows:
            where.setdefault(row["metadata"]["semantic_id"], set()).add(name)
    shared = {key: sorted(names) for key, names in where.items() if len(names) > 1}
    assert not shared, shared


def test_train_recipe_matches_doc() -> None:
    profiles = _yaml_profiles()
    recipes = _doc_profile_recipes()
    assert set(recipes) == set(profiles), f"PIPELINE.md recipe table {sorted(recipes)} != models.yaml {sorted(profiles)}"
    for name, profile in profiles.items():
        base, lr, warmup, batch, accum = recipes[name]
        assert (base, lr, warmup, batch, accum) == (
            Path(profile["trainer_model"]).name, profile["lr"], str(profile["warmup_ratio"]),
            str(profile["batch"]), str(profile["accum"]),
        ), f"PIPELINE.md recipe for {name} != models.yaml"
        recipe = SAYSO.train_recipe(name)
        assert Path(recipe[recipe.index("--model") + 1]).name == base
        assert recipe[recipe.index("--lr") + 1] == lr
        assert recipe[recipe.index("--warmup-ratio") + 1] == warmup


def test_release_constants_match_doc() -> None:
    assert SAYSO.RELEASE_SLACK == _doc_release_slack()
    assert tuple(SAYSO.RELEASE_SUITES) == _doc_release_suites()
    assert SAYSO.RELEASE_QUANT == _doc_release_quant()


def test_generation_config_is_versioned_file_under_documented_dir() -> None:
    doc_path = _doc_generation_config()
    assert SAYSO.GENERATION_CONFIG == doc_path
    config_path = SAYSO_TRAINING / SAYSO.GENERATION_CONFIG
    assert config_path.parent.relative_to(SAYSO_TRAINING).as_posix() == _doc_generation_dir().rstrip("/")
    assert config_path.is_file(), f"GENERATION_CONFIG file missing: {config_path}"


def test_rendered_training_view_has_no_tool_call_tags() -> None:
    pytest.importorskip("transformers")
    from adapters.lfm import expand_canonical_row_to_views
    from adapters.schema import contains_chatml_tool_call_markers

    views = [view for row in _load_jsonl(SMOKE_FIXTURE) for view in expand_canonical_row_to_views(row)]
    assert views, "smoke fixture renders no supervised views"
    for view in views:
        text = view["instruction"] + view["output"]
        assert "<tool_call>" not in text
        assert not contains_chatml_tool_call_markers(text)


def test_allowed_hass_tools_equals_v2_tool_names() -> None:
    if not SCHEMA_V2.is_file():
        pytest.skip(f"pinned schema not present (vendor/SaySo submodule unpopulated): {SCHEMA_V2}")
    from adapters.schema import ALLOWED_HASS_TOOLS, v2_tool_names

    assert ALLOWED_HASS_TOOLS == v2_tool_names()


def test_post_render_validation_rejects_eval_held_out_utterances() -> None:
    pytest.importorskip("evals.cases")
    from evals.cases import excluded_train_utterances
    from generators.validation import check_quality_eval_overlap

    held_out = sorted(excluded_train_utterances())
    assert held_out, "no eval-held-out utterances loaded"
    for utterance in held_out:
        assert check_quality_eval_overlap(utterance), utterance


def test_canary_and_train_run_through_gpu_lock() -> None:
    tree = ast.parse((SAYSO_TRAINING / "sayso").read_text(encoding="utf-8"))
    for name in ("cmd_canary", "cmd_train"):
        found = False
        for call in ast.walk(_function(tree, name)):
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "run" and call.args):
                continue
            command = call.args[0]
            if isinstance(command, ast.List) and len(command.elts) >= 2:
                first, second = command.elts[0], command.elts[1]
                if isinstance(first, ast.Name) and first.id == "GPU_LOCK" and isinstance(second, ast.Constant) and second.value == "train":
                    found = True
        assert found, f"{name} does not run the trainer through [GPU_LOCK, 'train', ...]"


def test_eval_compares_against_baseline_champion() -> None:
    assert SAYSO.BASELINE.resolve() == BASELINE.resolve()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    champion = baseline.get("eval", {}).get("champion", {})
    for key in ("name", "gguf", "gguf_sha256"):
        assert champion.get(key), f"training_baseline.json eval.champion.{key} missing"
    tree = ast.parse((SAYSO_TRAINING / "sayso").read_text(encoding="utf-8"))
    body = _function(tree, "cmd_eval")
    names = {node.id for node in ast.walk(body) if isinstance(node, ast.Name)}
    calls = {node.func.id for node in ast.walk(body) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "BASELINE" in names, "cmd_eval does not read the champion from BASELINE"
    assert "compare_release" in calls, "cmd_eval does not call compare_release"


GENERATION_INVARIANTS = {
    12: "Generation snapshots the working tree, never HEAD.",
    13: "Generation tests gate every candidate.",
    14: "Changes to gate files since the last promoted snapshot require `--accept-drift`.",
    15: "Generation records an input fingerprint.",
    16: "Promotion reproduces the candidate byte-for-byte.",
    17: "Full training refuses unless the candidate is reproduced and still matches.",
}


def _invariant_text(number: int) -> str:
    match = re.search(rf"^{number}\. \*\*.*?(?=^\d+\. \*\*|\Z)", INVARIANTS, re.M | re.S)
    assert match is not None, f"PIPELINE.md has no invariant {number}"
    return match.group(0)


def _definition(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} has no function {name}")


def _text(path: Path, name: str) -> str:
    return ast.get_source_segment(path.read_text(encoding="utf-8"), _definition(path, name)) or ""


def _called(node: ast.AST) -> set[str]:
    return {ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)}


def _dict_keys(node: ast.AST) -> set[str]:
    return {
        key.value
        for mapping in ast.walk(node) if isinstance(mapping, ast.Dict)
        for key in mapping.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


@pytest.mark.parametrize("number,title", sorted(GENERATION_INVARIANTS.items()))
def test_generation_invariant_is_listed_in_doc(number: int, title: str) -> None:
    assert re.match(rf"{number}\. \*\*{re.escape(title)}\*\*", _invariant_text(number)), (
        f"PIPELINE.md invariant {number} does not read {title!r}"
    )


def test_generate_snapshots_worktree_not_head() -> None:
    snapshot = _text(DATASET_GATES, "snapshot_worktree")
    assert "refs/sayso/snapshots/" in snapshot
    assert "commit-tree" in snapshot
    assert '"snapshot_commit": commit' in snapshot
    assert "HEAD" not in snapshot, "snapshot_worktree must not read HEAD"
    assert "snapshot_worktree" in _called(_definition(SAYSO_FILE, "cmd_generate"))


def test_generate_runs_fast_and_golden_tests_with_cache_and_drift_stop() -> None:
    body = _text(DATASET_GATES, "check_snapshot")
    assert '"training/tests", "-m", "not slow"' in body
    assert '"training/tests/test_generator_golden.py"' in body
    assert 'name == "fast" and result.returncode' in body, "fast-test failure must stop generation"
    assert "\"test-cache\"" in body and "snapshot['snapshot_tree']" in body
    assert "input_fingerprint" in body
    assert "if reason is None" in body, "golden drift must stop without --accept-drift"
    assert "check_snapshot" in _called(_definition(SAYSO_FILE, "cmd_generate"))


def test_protected_gate_paths_and_drift_guard() -> None:
    assert gates.GATE_PATHS == (
        "training/tests",
        "training/generators/validation.py",
        "training/preflight.py",
        "evals/config/gates.yaml",
        "training/dataset_gates.py",
        "sayso",
    )
    body = _text(DATASET_GATES, "check_gate_changes")
    assert "gate-diff.patch" in body
    assert "if reason is None" in body
    assert "check_gate_changes" in _called(_definition(SAYSO_FILE, "cmd_generate"))
    invariant = _invariant_text(14)
    for path in gates.GATE_PATHS:
        assert f"`{path}`" in invariant, f"invariant 14 does not list {path}"


def test_input_fingerprint_covers_required_inputs() -> None:
    required = {
        "generator_tree", "recipe_sha256", "config_files", "eval_cases", "sayso_commit",
        "python_version", "packages", "tokenizer",
    }
    assert required <= _dict_keys(_definition(DATASET_GATES, "input_fingerprint"))
    assert '"input_fingerprint": fingerprint' in _text(SAYSO_FILE, "cmd_generate")


def test_promote_dataset_regenerates_in_snapshot_and_compares_bytes() -> None:
    node = _definition(SAYSO_FILE, "cmd_promote_dataset")
    assert {"materialize_snapshot", "generate_dataset"} <= _called(node)
    assert "candidate_sha256 != reproduced_sha256" in _text(SAYSO_FILE, "cmd_promote_dataset")


def test_train_checks_candidate_before_gpu_run() -> None:
    check = _text(DATASET_GATES, "check_trainable")
    assert 'record.get("reproduced") is not True' in check
    assert "reproduced_sha256" in check
    assert "input fingerprint mismatch" in check
    assert {"materialize_snapshot", "fingerprint_snapshot"} <= _called(_definition(DATASET_GATES, "check_trainable"))
    assert "160000" in check and "generator_tree" in check and "sayso_commit" in check
    train = _definition(SAYSO_FILE, "cmd_train")
    checks = [
        call.lineno for call in ast.walk(train)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "check_trainable"
    ]
    gpu_runs = [
        call.lineno for call in ast.walk(train)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "run" and call.args
        and isinstance(call.args[0], ast.List) and call.args[0].elts
        and isinstance(call.args[0].elts[0], ast.Name) and call.args[0].elts[0].id == "GPU_LOCK"
    ]
    assert checks and gpu_runs and max(checks) < min(gpu_runs), "check_trainable must run before the GPU lock"
