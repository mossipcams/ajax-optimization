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

TESTS = Path(__file__).resolve().parent
TRAINING = TESTS.parent
SAYSO_TRAINING = TRAINING.parent
DOCS = SAYSO_TRAINING / "docs" / "PIPELINE.md"
TRAINER = TRAINING / "scripts" / "train_unsloth_full.py"
SPLIT_SCRIPT = TRAINING / "scripts" / "split_dataset.py"
SMOKE_FIXTURE = TRAINING / "fixtures" / "sayso_lfm_smoke.jsonl"
BASELINE = TRAINING / "configs" / "training_baseline.json"
SCHEMA_V2 = SAYSO_TRAINING / "schemas" / "sayso-tool-schema-v2.json"

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


def _doc_base_model() -> str:
    return _doc_match(r"Base model is `([A-Za-z0-9._\-]+)`", INVARIANTS, "the base model").group(1)


def _doc_val_ratio() -> float:
    return float(_doc_match(r"VAL_RATIO = ([0-9.]+)", INVARIANTS, "VAL_RATIO").group(1))


def _doc_generation_dir() -> str:
    return _doc_match(r"under `(training/configs/generation/)`", INVARIANTS, "the generation config directory").group(1)


def _doc_generation_config() -> str:
    return _doc_match(
        r"`(training/configs/generation/[^`]+)`,?\s+named by the\s+`GENERATION_CONFIG`", DOC_TEXT, "GENERATION_CONFIG"
    ).group(1)


def _doc_train_recipe() -> tuple[str, str]:
    match = _doc_match(r"`--lr ([0-9.e\-]+) --warmup-ratio ([0-9.]+)`\s*\(`TRAIN_RECIPE`", DOC_TEXT, "TRAIN_RECIPE")
    return match.group(1), match.group(2)


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


def test_base_model_matches_doc_in_sayso_and_trainer() -> None:
    doc_name = _doc_base_model()
    assert doc_name == "LFM2.5-230M-Base"
    assert Path(SAYSO.BASE_MODEL).name == doc_name, f"sayso BASE_MODEL {SAYSO.BASE_MODEL!r} != doc {doc_name!r}"
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    defaults = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--model":
                defaults.extend(_joined_constants(kw.value) for kw in node.keywords if kw.arg == "default")
    assert defaults, "train_unsloth_full.py has no --model default"
    for default in defaults:
        assert default.rsplit("/", 1)[-1] == doc_name, f"--model default {default!r} != doc {doc_name!r}"


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def test_trainer_kwargs_full_finetuning_bf16() -> None:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
            calls.setdefault(ast.unparse(node.func), []).append(kwargs)
    fast_model = calls.get("FastModel.from_pretrained", [])
    training_args = calls.get("TrainingArguments", [])
    assert any(_is_true(kwargs.get("full_finetuning")) for kwargs in fast_model), "FastModel(full_finetuning=True) missing"
    assert any(ast.unparse(kwargs.get("dtype", "")) == "torch.bfloat16" for kwargs in fast_model), (
        "FastModel(dtype=torch.bfloat16) missing"
    )
    assert not any(_is_true(kwargs.get("float32_mixed_precision")) for kwargs in fast_model), (
        "FastModel(float32_mixed_precision=True) must not be set"
    )
    assert any(_is_true(kwargs.get("bf16")) for kwargs in training_args), "TrainingArguments(bf16=True) missing"


def test_trainer_has_no_adapter_path() -> None:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
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
    doc_lr, doc_warmup = _doc_train_recipe()
    assert SAYSO.TRAIN_RECIPE[SAYSO.TRAIN_RECIPE.index("--lr") + 1] == doc_lr
    assert SAYSO.TRAIN_RECIPE[SAYSO.TRAIN_RECIPE.index("--warmup-ratio") + 1] == doc_warmup


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
