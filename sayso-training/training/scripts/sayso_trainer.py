#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_common import (
    HOST,
    load_eval_rows,
    load_rows,
    package_versions,
    preflight,
    recipe,
    sha256,
    stop_callback,
)
from train_common import build_manifest as common_build_manifest
from train_common import build_parser as common_build_parser
from train_common import parse_args as common_parse_args

FLATTENING_KEYS = ("seq_idx", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "position_ids")


def load_pins() -> dict[str, str]:
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        name, _, pinned = line.strip().partition("==")
        pins[name] = pinned
    return pins


REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements-sayso-trainer.txt"
PINS = load_pins()


def check_stack(versions: dict[str, str | None]) -> None:
    found = {name: versions.get(name) for name in PINS}
    if found != PINS:
        sys.exit(f"sayso-trainer stack found {found} under {sys.executable}; pinned {PINS}; "
                 f"rebuild {sys.prefix} with scripts/llm-host/sayso-trainer-venv.sh")


def check_installed_stack() -> None:
    check_stack(package_versions(tuple(PINS)))


def check_batch(batch) -> None:
    shape = tuple(batch["input_ids"].shape)
    if len(shape) != 2:
        raise ValueError(f"batch input_ids shape {shape} is not 2-D")
    if "attention_mask" not in batch:
        raise ValueError("batch lacks attention_mask; a padded batch must carry its mask")
    flattening = [key for key in FLATTENING_KEYS if key in batch]
    if flattening:
        raise ValueError(f"batch carries flattening keys {flattening}; padded batches only")


def sft_recipe(args, evaluating: bool) -> dict:
    training = recipe(args, evaluating=evaluating)
    training["warmup_steps"] = training.pop("warmup_ratio")
    training["remove_unused_columns"] = False
    return training


def build_manifest(args, training_args: dict, data_sha256: str) -> dict:
    manifest = common_build_manifest(args, training_args, data_sha256, backend="hf",
                                     trainer_sha256=sha256(__file__))
    manifest["method"] = getattr(args, "method", "sft")
    return manifest


def parse_args(argv: list[str] | None = None):
    parser = common_build_parser()
    parser.add_argument("--model", default=f"{HOST}/lfm/models/LFM2.5-230M-Base")
    parser.add_argument("--method", choices=("sft",), default="sft")
    return common_parse_args(argv, parser=parser, run_name="sayso-trainer")


def main() -> int:
    args = parse_args()
    check_installed_stack()
    print(f"sayso-trainer method={args.method} batch={args.batch} accum={args.accum}", flush=True)
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        sys.exit("no HIP device")
    print(f"torch {torch.__version__} hip={torch.version.hip} device={torch.cuda.get_device_name(0)}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        sys.exit("tokenizer has no pad_token_id; set one before training")

    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    dataset = load_rows(args.data, tokenizer)
    eval_dataset = load_eval_rows(args.eval_data, tokenizer, args.cutoff) if args.eval_data else None
    preflight(dataset, args.cutoff)
    dataset = dataset.remove_columns(["length", "supervised"])
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        processing_class=tokenizer,
        args=SFTConfig(
            bf16=True,
            gradient_checkpointing=True,
            max_length=args.cutoff,
            packing=False,
            padding_free=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            **sft_recipe(args, evaluating=eval_dataset is not None),
        ),
        callbacks=[stop_callback(args.stop_at_step)] if args.stop_at_step else None,
    )
    probe = trainer.data_collator([dataset[i] for i in range(min(args.batch, len(dataset)))])
    check_batch(probe)

    manifest = build_manifest(args, trainer.args.to_dict(), sha256(args.data))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    result = trainer.train()
    rates = [entry["learning_rate"] for entry in trainer.state.log_history if "learning_rate" in entry]
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), final_step=trainer.state.global_step,
                    max_steps=trainer.state.max_steps, last_logged_learning_rate=rates[-1] if rates else None,
                    eval_loss=[{"step": entry["step"], "eval_loss": entry["eval_loss"]}
                               for entry in trainer.state.log_history if "eval_loss" in entry],
                    parameter_dtypes=sorted({str(param.dtype) for param in model.parameters() if param.requires_grad}),
                    optimizer_state_dtypes=sorted({str(state["exp_avg"].dtype)
                                                   for state in trainer.optimizer.state.values() if "exp_avg" in state}),
                    train_metrics=result.metrics, peak_vram_bytes=torch.cuda.max_memory_allocated())
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    trainer.save_model(f"{args.out}/final")
    tokenizer.save_pretrained(f"{args.out}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
