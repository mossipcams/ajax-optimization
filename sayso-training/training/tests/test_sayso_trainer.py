from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("sayso_trainer", SCRIPTS / "sayso_trainer.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)
COMMON = sys.modules["train_common"]
UNSLOTH_SPEC = importlib.util.spec_from_file_location("train_unsloth_full", SCRIPTS / "train_unsloth_full.py")
UNSLOTH = importlib.util.module_from_spec(UNSLOTH_SPEC)
UNSLOTH_SPEC.loader.exec_module(UNSLOTH)
DOCS = Path(__file__).resolve().parents[2] / "docs" / "PIPELINE.md"


def _tensor(*shape: int) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


def test_backend_flag_is_gone() -> None:
    with pytest.raises(SystemExit):
        TRAIN.parse_args(["--canary", "--out", "/tmp/sayso-trainer-canary", "--backend", "unsloth"])


def test_defaults_give_batch_8_and_accum_4() -> None:
    args = TRAIN.parse_args(["--canary", "--out", "/tmp/sayso-trainer-canary"])
    assert args.batch == 8
    assert args.accum == 4


def test_canary_keeps_the_full_run_schedule_and_stops_early() -> None:
    args = TRAIN.parse_args(["--canary", "--out", "/tmp/sayso-trainer-canary"])
    assert args.max_steps == -1
    assert args.stop_at_step == 250


def test_full_training_without_promotion_record_exits(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    with pytest.raises(SystemExit):
        TRAIN.parse_args(["--data", str(data), "--out", str(tmp_path / "full")])


def test_guard_rejects_non_2d_or_wrong_batch_dimension() -> None:
    for shape in ((1, 16), (8,), (4, 16)):
        with pytest.raises(ValueError):
            TRAIN.check_batch({"input_ids": _tensor(*shape), "attention_mask": _tensor(*shape)}, 8)


def test_guard_rejects_flattening_keys() -> None:
    batch = {"input_ids": _tensor(8, 16), "attention_mask": _tensor(8, 16), "cu_seq_lens_q": _tensor(9)}
    with pytest.raises(ValueError):
        TRAIN.check_batch(batch, 8)


def test_guard_accepts_padded_batches() -> None:
    batch = {"input_ids": _tensor(8, 16), "attention_mask": _tensor(8, 16), "labels": _tensor(8, 16)}
    TRAIN.check_batch(batch, 8)


def test_guard_rejects_position_ids_without_attention_mask() -> None:
    batch = {"input_ids": _tensor(8, 16), "labels": _tensor(8, 16), "position_ids": _tensor(8, 16)}
    with pytest.raises(ValueError):
        TRAIN.check_batch(batch, 8)


def test_encode_is_the_train_common_encode() -> None:
    assert TRAIN.encode is COMMON.encode
    assert TRAIN.load_rows is COMMON.load_rows
    assert UNSLOTH.load_rows is COMMON.load_rows


def test_manifest_records_backend_and_train_common_hash(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    args = SimpleNamespace(canary=True, stop_at_step=250, model=str(model),
                           data="/data.jsonl", eval_data=None)
    manifest = TRAIN.build_manifest(args, {"seed": 1}, "data-sha")
    assert manifest["backend"] == "hf"
    assert manifest["train_common_sha256"] == TRAIN.sha256(SCRIPTS / "train_common.py")
    assert manifest["trainer_sha256"] == TRAIN.sha256(SCRIPTS / "sayso_trainer.py")
    assert "trl" in manifest["packages"]


def test_sayso_trainer_uses_the_shared_recipe() -> None:
    assert TRAIN.recipe is COMMON.recipe


def test_recipe_gives_micro_batch_group_by_length_and_no_bf16() -> None:
    args = TRAIN.parse_args(["--canary", "--out", "/tmp/sayso-trainer-canary"])
    training = TRAIN.recipe(args, evaluating=True)
    assert training["per_device_train_batch_size"] == 8
    assert training["gradient_accumulation_steps"] == 4
    assert training["per_device_eval_batch_size"] == 1
    assert training["train_sampling_strategy"] == "group_by_length"
    assert training["eval_strategy"] == "steps"
    assert "bf16" not in training


def test_sayso_trainer_config_sets_bf16_and_disables_packing_and_padding_free() -> None:
    tree = ast.parse((SCRIPTS / "sayso_trainer.py").read_text(encoding="utf-8"))
    configs = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and ast.unparse(node.func) == "SFTConfig"]
    assert len(configs) == 1
    kwargs = {kw.arg: kw.value for kw in configs[0].keywords}
    assert isinstance(kwargs["bf16"], ast.Constant) and kwargs["bf16"].value is True
    assert isinstance(kwargs["packing"], ast.Constant) and kwargs["packing"].value is False
    assert isinstance(kwargs["padding_free"], ast.Constant) and kwargs["padding_free"].value is False


def _joined_constants(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return ""


def _model_defaults(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--model":
                defaults.extend(_joined_constants(kw.value) for kw in node.keywords if kw.arg == "default")
    return defaults


def _doc_base_model() -> str:
    text = DOCS.read_text(encoding="utf-8")
    invariants = re.search(r"^## Pipeline invariants\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert invariants is not None, "PIPELINE.md has no 'Pipeline invariants' section"
    match = re.search(r"Base model is `([A-Za-z0-9._\-]+)`", invariants.group(1), re.S)
    assert match is not None, "PIPELINE.md does not state the base model"
    return match.group(1)


def test_model_default_matches_unsloth_trainer_and_doc_base_model() -> None:
    defaults = _model_defaults(SCRIPTS / "sayso_trainer.py")
    assert defaults, "sayso_trainer.py has no --model default"
    assert defaults == _model_defaults(SCRIPTS / "train_unsloth_full.py")
    doc_name = _doc_base_model()
    for default in defaults:
        assert default.rsplit("/", 1)[-1] == doc_name, f"--model default {default!r} != doc {doc_name!r}"


def test_model_is_loaded_with_automodel_in_bf16() -> None:
    tree = ast.parse((SCRIPTS / "sayso_trainer.py").read_text(encoding="utf-8"))
    loads = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "AutoModelForCausalLM.from_pretrained"]
    assert len(loads) == 1
    kwargs = {kw.arg: kw.value for kw in loads[0].keywords}
    assert "dtype" in kwargs, "AutoModelForCausalLM.from_pretrained has no dtype"
    assert ast.unparse(kwargs["dtype"]) == "torch.bfloat16"


def test_trainer_imports_no_peft_and_no_fp32_master_weights() -> None:
    source = (SCRIPTS / "sayso_trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "peft" not in imported
    for banned in ("get_peft_model", "LoraConfig", "float32_mixed_precision"):
        assert banned not in source, banned
