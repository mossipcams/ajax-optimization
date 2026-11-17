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
    CANARY_STEPS,
    HOST,
    build_parser,
    load_eval_rows,
    load_rows,
    preflight,
    recipe,
    sha256,
    stop_callback,
)
from train_common import build_manifest as common_build_manifest
from train_common import parse_args as common_parse_args


def build_manifest(args, training_args: dict, data_sha256: str) -> dict:
    return common_build_manifest(args, training_args, data_sha256, backend="unsloth", trainer_sha256=sha256(__file__))


def training_arguments(args, evaluating: bool):
    from transformers import TrainingArguments

    return TrainingArguments(bf16=True, **recipe(args, evaluating))


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    parser.add_argument("--model", default=f"{HOST}/lfm/models/LFM2.5-230M-Base")
    return common_parse_args(argv, parser=parser)


def main() -> int:
    args = parse_args()
    from unsloth import FastModel
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer

    print(f"torch {torch.__version__} hip={torch.version.hip} device={torch.cuda.get_device_name(0)}")
    if not torch.version.hip or not torch.cuda.is_available():
        sys.exit("no HIP device")
    model, tokenizer = FastModel.from_pretrained(
        args.model,
        max_seq_length=args.cutoff,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,
    )
    # LFM2 short-conv layers are causal and unmasked, so pads must trail the real tokens.
    tokenizer.padding_side = "right"
    assert tokenizer.pad_token_id is not None, "tokenizer has no pad token; right padding needs one"

    dataset = load_rows(args.data, tokenizer)
    eval_dataset = load_eval_rows(args.eval_data, tokenizer, args.cutoff) if args.eval_data else None
    preflight(dataset, args.cutoff)
    dataset = dataset.remove_columns(["length", "supervised"])
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        processing_class=tokenizer,
        args=training_arguments(args, evaluating=eval_dataset is not None),
        callbacks=[stop_callback(args.stop_at_step)] if args.stop_at_step else None,
    )
    manifest = build_manifest(args, trainer.args.to_dict(), sha256(args.data))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    result = trainer.train()
    peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    rates = [entry["learning_rate"] for entry in trainer.state.log_history if "learning_rate" in entry]
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), final_step=trainer.state.global_step,
                    max_steps=trainer.state.max_steps, train_metrics=result.metrics, peak_vram_bytes=peak_vram_bytes,
                    last_logged_learning_rate=rates[-1] if rates else None,
                    eval_loss=[{"step": entry["step"], "eval_loss": entry["eval_loss"]}
                               for entry in trainer.state.log_history if "eval_loss" in entry],
                    parameter_dtypes=sorted({str(param.dtype) for param in model.parameters() if param.requires_grad}),
                    optimizer_state_dtypes=sorted({str(state["exp_avg"].dtype)
                                                   for state in trainer.optimizer.state.values() if "exp_avg" in state}))
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    trainer.save_model(f"{args.out}/final")
    tokenizer.save_pretrained(f"{args.out}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
