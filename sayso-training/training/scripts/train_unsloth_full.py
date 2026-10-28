#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HOST = "/workspace/host"
CANARY_STEPS = 250


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def package_versions(names=("torch", "transformers", "unsloth", "datasets", "accelerate")) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(args, training_args: dict, data_sha256: str) -> dict:
    return {
        "argv": sys.argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "canary" if args.canary else "full",
        "stop_at_step": args.stop_at_step,
        "trainer_sha256": sha256(__file__),
        "base_model": args.model,
        "base_model_files": tree_sha256(args.model),
        "data": args.data,
        "data_sha256": data_sha256,
        "training_args": training_args,
        "packages": package_versions(),
    }


def stop_callback(stop_at_step: int):
    from transformers import TrainerCallback

    class StopAtStep(TrainerCallback):
        """End a run early while keeping the full run's LR horizon and warmup."""

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_at_step:
                control.should_training_stop = True
            return control

    return StopAtStep()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=f"{HOST}/lfm/models/LFM2.5-230M-Base")
    parser.add_argument("--data", default=f"{HOST}/datasets/sayso_full_sft_v5_20260922_rendered.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--cutoff", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--canary", action="store_true",
                        help=f"Run the first {CANARY_STEPS} optimizer steps of the full-run schedule")
    parser.add_argument("--stop-at-step", type=int, default=0,
                        help="stop after this optimizer step without shortening the LR schedule")
    parser.add_argument("--promotion-record", help="Required promotion metadata for full training")
    args = parser.parse_args(argv)
    print(f"argv={sys.argv} lr={args.lr} warmup_ratio={args.warmup_ratio}", flush=True)
    if args.canary:
        args.stop_at_step = args.stop_at_step or CANARY_STEPS
    elif not args.promotion_record:
        parser.error("full training requires --promotion-record from ./sayso promote-dataset")
    else:
        with open(args.promotion_record, encoding="utf-8") as handle:
            promotion = json.load(handle)
        if (not promotion.get("promoted") or not promotion.get("validation_passed")
                or not promotion.get("canary_passed")
                or sha256(args.data) != promotion.get("rendered_sha256")):
            parser.error("full training blocked: data does not match the explicitly promoted dataset")
    args.out = args.out or (
        f"{HOST}/lfm/runs/sayso-lfm-v5-canary" if args.canary else f"{HOST}/lfm/runs/sayso-lfm-v5-full"
    )
    return args


def main() -> int:
    args = parse_args()
    from unsloth import FastModel
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

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

    def encode(row):
        prompt = tokenizer(row["instruction"], add_special_tokens=False)["input_ids"]
        completion = tokenizer(row["output"], add_special_tokens=False)["input_ids"]
        return {
            "input_ids": prompt + completion,
            "attention_mask": [1] * (len(prompt) + len(completion)),
            "labels": [-100] * len(prompt) + completion,
            "length": len(prompt) + len(completion),
            "supervised": len(completion),
        }

    def rows(path, stamp):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield {"instruction": row["instruction"], "output": row["output"]}

    stat = os.stat(args.data)
    dataset = Dataset.from_generator(
        rows, gen_kwargs={"path": args.data, "stamp": f"{stat.st_size}:{stat.st_mtime_ns}"}
    )
    dataset = dataset.map(encode, remove_columns=dataset.column_names, num_proc=6)
    lengths = dataset["length"]
    too_long = sum(length > args.cutoff for length in lengths)
    empty = sum(count == 0 for count in dataset["supervised"])
    print(
        f"rows={len(dataset)} max_tokens={max(lengths)} "
        f"supervised_tokens={sum(dataset['supervised'])} over_cutoff={too_long} empty_completion={empty}"
    )
    if too_long or empty:
        sys.exit("preflight failed")
    dataset = dataset.remove_columns(["length", "supervised"])
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            weight_decay=0.0,
            bf16=True,
            optim="adamw_torch",
            logging_steps=5,
            save_strategy="steps",
            save_steps=200,
            # ponytail: weights only (~0.5 GB each) so every checkpoint survives for ./sayso select;
            # intermediate checkpoints can't be resumed. Keep optimizer state if resuming is ever needed.
            save_total_limit=None,
            save_only_model=True,
            report_to="none",
            seed=20260922,
        ),
        callbacks=[stop_callback(args.stop_at_step)] if args.stop_at_step else None,
    )
    manifest = build_manifest(args, trainer.args.to_dict(), sha256(args.data))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    trainer.train()
    rates = [entry["learning_rate"] for entry in trainer.state.log_history if "learning_rate" in entry]
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), final_step=trainer.state.global_step,
                    max_steps=trainer.state.max_steps, last_logged_learning_rate=rates[-1] if rates else None)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    trainer.save_model(f"{args.out}/final")
    tokenizer.save_pretrained(f"{args.out}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
