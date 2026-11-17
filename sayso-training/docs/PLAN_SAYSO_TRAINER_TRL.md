# SaySo Trainer on TRL: replace the Unsloth trainer (plan)

Date: 2026-11-17. Base: `0811426`. PR #39 (`a7e78e1`) merged first; this branch
sits on top of it.

Goal: make the LFM trainer a replaceable component behind a fixed contract,
implemented on TRL. `sayso` already orchestrates (validate, canary, promote,
pin, eval), so it needs a library under it, not a framework. TRL is plain
PyTorch, trains any `AutoModelForCausalLM` model, and is what Axolotl and
LlamaFactory wrap, so ROCm now and CUDA later need no rewrite.

## Contract (unchanged)

- In: rendered JSONL (instruction/output), base model dir, recipe flags, and a
  promotion record (full runs only).
- Out: `<out>/checkpoint-*` (weights only), `<out>/final`, `<out>/manifest.json`.
- Preflight: zero truncation, no empty completions.
- Canary: first N optimizer steps of the full schedule (`sayso canary` passes
  `--canary`).

## Status (2026-11-17)

- **0. Environment check.** Done. The `unsloth` container
  (`unsloth/unsloth-rocm:studio`, pulled image, no build in this repo) has
  trl 0.24.0, transformers 5.5.0, torch 2.14.0+rocm7.2, and unsloth 2026.9.7.
- **1. Shared code.** Done. `training/scripts/train_common.py` holds hashing,
  package versions (now with trl), the manifest (now with `backend` and
  `train_common_sha256`), the stop callback, the parser and promotion gate
  (including `--batch 8` / `--accum 4`), encode/load_rows/load_eval_rows/
  preflight, and `recipe()`, which is every TrainingArguments kwarg except bf16.
  Each trainer keeps its own literal config call with `bf16=True`, so the
  precision invariant stays visible per trainer.
- **2. New trainer.** Done as `training/scripts/sayso_trainer.py` (the SaySo
  Trainer): TRL SFTTrainer on transformers, with the Unsloth backend dropped.
  It loads `AutoModelForCausalLM` in bf16 with sdpa, gradient checkpointing on,
  `use_cache` off, and right padding. The manifest records backend `hf`. The
  batch-shape guard stays: it pulls one batch before training and restores the
  torch RNG state, so sample order is unchanged.
- **3. CPU tests.** Done. `training/tests/test_sayso_trainer.py` covers the CLI,
  the guard, shared encode/load_rows/recipe identity, the manifest, and AST
  invariants (the `--model` default, `AutoModelForCausalLM` in bf16,
  `SFTConfig` with bf16 / `packing=False` / `padding_free=False`, and no
  PEFT/LoRA/fp32 master weights).
- **4. GPU canary A/B.** Skipped by user decision ("not worth comparing"). The
  planned A/B (A `train_unsloth_full.py` 8x4, B hf, C unsloth) was not run, and
  no GPU run of `sayso_trainer.py` has happened. The planned gates were
  eval_loss@200 within 2%, identical smoke gates, dev within 2 cases, and
  throughput no more than 10% worse. Estimated cost had it run: ~1.5-2.2 h GPU.
- **5. Wire into `sayso`.** Done. `sayso canary` and `sayso train` call
  `sayso_trainer.py`. `train_unsloth_full.py` stays as the rollback trainer until
  the first full run on `sayso_trainer.py` passes `./sayso eval` against the
  champion. `test_pipeline_invariants.py` still targets
  `train_unsloth_full.py`; existing tests were not edited.
- **6. First full run vs champion.** Pending. The next promoted dataset trains
  through `./sayso train` and is compared with the champion.

## Decisions

- **Unsloth backend dropped.** User decision. The SaySo Trainer does not use
  Unsloth; `train_unsloth_full.py` is kept only for rollback.
- **GPU A/B skipped.** User decision ("not worth comparing"). The first
  `./sayso canary` is the first real run.
- **TRL 0.24 target.** The container ships trl 0.24.0 with transformers 5.5.0.
  TRL is below 1.0 and there is no container build to pin it in, so the trainer
  targets 0.24, where `SFTConfig` `max_length`, `packing`, `padding_free`, and
  `dataset_kwargs` `skip_prepare_dataset` exist. transformers 5.x removed
  `TrainingArguments.group_by_length`, so the recipe uses
  `train_sampling_strategy="group_by_length"`.

## Rollback

Point the two `run()` calls in `sayso` (canary and train) back at
`training/scripts/train_unsloth_full.py`.

## Later work (separate plans)

- TRL native prompt/completion format, with a token-parity check.
- Packing or padding-free on LFM2, once a `seq_idx` conv reset is verified on ROCm.
- DPOTrainer / GRPOTrainer behind the same contract. `--method` (default `sft`) is the slot for them.
- `trl sft --config` YAML recipes.
