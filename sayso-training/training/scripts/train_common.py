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
EVAL_ROWS = 512
SEED = 20260922


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def package_versions(names=("torch", "transformers", "unsloth", "datasets", "accelerate", "trl", "tokenizers")) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(args, training_args: dict, data_sha256: str, *, backend: str, trainer_sha256: str) -> dict:
    return {
        "argv": sys.argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "mode": "canary" if args.canary else "full",
        "stop_at_step": args.stop_at_step,
        "trainer_sha256": trainer_sha256,
        "train_common_sha256": sha256(__file__),
        "base_model": args.model,
        "base_model_files": tree_sha256(args.model),
        "data": args.data,
        "data_sha256": data_sha256,
        "eval_data": getattr(args, "eval_data", None),
        "training_args": training_args,
        "packages": package_versions(),
    }


def stop_callback(stop_at_step: int):
    from transformers import TrainerCallback

    class StopAtStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_at_step:
                control.should_training_stop = True
            return control

    return StopAtStep()


def recipe(args, evaluating: bool) -> dict:
    return {
        "output_dir": args.out,
        "per_device_train_batch_size": args.batch,
        "gradient_accumulation_steps": args.accum,
        "train_sampling_strategy": "group_by_length",
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.0,
        "optim": "adamw_torch",
        "logging_steps": 5,
        "eval_strategy": "steps" if evaluating else "no",
        "eval_steps": 200,
        "per_device_eval_batch_size": 1,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": None,
        "save_only_model": True,
        "report_to": "none",
        "seed": SEED,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=f"{HOST}/datasets/sayso_full_sft_v5_20260922_rendered.jsonl")
    parser.add_argument("--eval-data", help="group-held-out rendered rows for validation loss")
    parser.add_argument("--out")
    parser.add_argument("--cutoff", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch", type=int, default=1, help="per-device micro-batch size")
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--canary", action="store_true",
                        help=f"Run the first {CANARY_STEPS} optimizer steps of the full-run schedule")
    parser.add_argument("--stop-at-step", type=int, default=0,
                        help="stop after this optimizer step without shortening the LR schedule")
    parser.add_argument("--promotion-record", help="Required promotion metadata for full training")
    return parser


def parse_args(argv: list[str] | None = None, parser: argparse.ArgumentParser | None = None,
               run_name: str = "sayso-lfm-v5") -> argparse.Namespace:
    if parser is None:
        parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.warmup_ratio < 1:
        parser.error(f"--warmup-ratio {args.warmup_ratio} must be in [0, 1)")
    print(f"argv={sys.argv} lr={args.lr} warmup_ratio={args.warmup_ratio}", flush=True)
    print(f"effective_batch = batch * accum = {args.batch} * {args.accum} = {args.batch * args.accum}", flush=True)
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
        f"{HOST}/lfm/runs/{run_name}-canary" if args.canary else f"{HOST}/lfm/runs/{run_name}-full"
    )
    return args


def tokenize_view(row, tokenizer):
    prompt = tokenizer(row["instruction"], add_special_tokens=False)["input_ids"]
    completion = tokenizer(row["output"], add_special_tokens=False)["input_ids"]
    return prompt, completion


def encode(row, tokenizer):
    prompt, completion = tokenize_view(row, tokenizer)
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


def view_groups(path):
    seen = set()
    group, source = [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = row["metadata"]["source_row_index"]
            if index != source:
                if group:
                    yield source, group
                if index in seen:
                    raise ValueError(f"source_row_index {index} reappears non-adjacently in {path}; "
                                     "--merge-views needs each conversation's views adjacent")
                seen.add(index)
                group, source = [], index
            group.append({"instruction": row["instruction"], "output": row["output"]})
    if group:
        yield source, group


def merge_group(views, tokenizer, source):
    if len(views) == 1:
        return encode(views[0], tokenizer)
    tokenized = [tokenize_view(view, tokenizer) for view in views]
    full = max((prompt + completion for prompt, completion in tokenized), key=len)
    labels = [-100] * len(full)
    for prompt, completion in tokenized:
        ids = prompt + completion
        if full[:len(ids)] != ids:
            raise ValueError(f"source_row_index {source}: a sibling view is not an exact token prefix of the "
                             "longest view; --merge-views cannot merge it")
        labels[len(prompt):len(ids)] = completion
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": labels,
        "length": len(full),
        "supervised": sum(label != -100 for label in labels),
    }


def merged_rows(path, stamp, tokenizer):
    for source, views in view_groups(path):
        yield merge_group(views, tokenizer, source)


def load_rows(path, tokenizer, merge_views=False):
    from datasets import Dataset

    stat = os.stat(path)
    stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    if merge_views:
        return Dataset.from_generator(merged_rows, gen_kwargs={"path": path, "stamp": stamp, "tokenizer": tokenizer})
    dataset = Dataset.from_generator(rows, gen_kwargs={"path": path, "stamp": stamp})
    return dataset.map(encode, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dataset.column_names, num_proc=6)


def load_eval_rows(path, tokenizer, cutoff: int, merge_views=False):
    eval_dataset = load_rows(path, tokenizer, merge_views).shuffle(seed=SEED)
    eval_dataset = eval_dataset.select(range(min(EVAL_ROWS, len(eval_dataset))))
    eval_dataset = eval_dataset.filter(lambda row: row["length"] <= cutoff and row["supervised"] > 0)
    return eval_dataset.remove_columns(["length", "supervised"])


def preflight(dataset, cutoff: int) -> None:
    lengths = dataset["length"]
    too_long = sum(length > cutoff for length in lengths)
    empty = sum(count == 0 for count in dataset["supervised"])
    print(
        f"rows={len(dataset)} max_tokens={max(lengths)} "
        f"supervised_tokens={sum(dataset['supervised'])} over_cutoff={too_long} empty_completion={empty}"
    )
    if too_long or empty:
        sys.exit("preflight failed")
