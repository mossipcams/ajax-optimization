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
  (`unsloth/unsloth-rocm:studio`, pulled image, no build in this repo) keeps
  trl 0.24.0, transformers 5.5.0, torch 2.14.0+rocm7.2, and unsloth 2026.9.7
  for the rollback trainer. The SaySo Trainer runs on trl 1.18.1 in its own
  venv inside that container (host `/srv/llm/lfm/venvs/sayso-trainer`, created
  by `sayso-training/scripts/llm-host/sayso-trainer-venv.sh`) with the exact
  pins in `sayso-training/training/requirements-sayso-trainer.txt` (trl 1.18.1,
  transformers 5.23.0, datasets 5.2.0, tokenizers 0.23.3; Renovate-managed).
  `sayso_trainer.check_installed_stack()` requires those exact versions. A `.pth` reuses the
  container's torch 2.14.0+rocm7.2 and accelerate 1.15.0.
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
  batch-shape guard stays: before training it runs the trainer's collator on the first
  rows and requires 2-D input_ids, an attention_mask, and no flattening keys
  (seq_idx, cu_seq_lens_*, max_length_*, position_ids).
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
  `sayso_trainer.py` through `TRAINER_PYTHON` (env `SAYSO_TRAINER_PYTHON`).
  `train_unsloth_full.py` and its test were deleted on 2026-11-18, so the
  active trainer is the only trainer. `test_pipeline_invariants.py` checks
  `sayso_trainer.py`.
- **6. First full run vs champion.** Pending. The next promoted dataset trains
  through `./sayso train` and is compared with the champion.

## Decisions

- **Unsloth backend dropped.** User decision. The SaySo Trainer does not use
  Unsloth; `train_unsloth_full.py` and its test were deleted on 2026-11-18.
- **GPU A/B skipped.** User decision ("not worth comparing"). The first
  `./sayso canary` is the first real run.
- **TRL 1.18.1 in its own venv.** The container's trl 0.24.0 / transformers 5.5.0
  are left as they are. Unsloth 2026.9.x caps trl at 1.13 and
  transformers at 5.17, so the SaySo Trainer gets a separate venv rather than
  the container's Python. `sayso` launches it through `TRAINER_PYTHON`, which
  defaults to that venv's `bin/python`. transformers 5.x removed
  `TrainingArguments.group_by_length`, so the recipe uses
  `train_sampling_strategy="group_by_length"`. Two compatibility changes:
  - transformers 5.23 removed `TrainingArguments.warmup_ratio`, so `sft_recipe`
    maps `--warmup-ratio` to a float `warmup_steps` in [0, 1). The result is the
    same `ceil(total * ratio)`.
  - TRL 1.18's SFTTrainer keeps only `input_ids`, `labels`, and `seq_lengths` by
    default, so the trainer passes `remove_unused_columns=False` to keep
    `attention_mask`, which the batch guard requires.
  TRL 1.18's `SFTConfig` defaults `gradient_checkpointing=True`, which matches
  the trainer.

## Rollback

Revert PR #49, then PR #41, then PR #40 (`git revert -m 1 <merge commit>` of each, newest first).
Reverting the deletion commit (`chore(sayso-training): delete the Unsloth rollback trainer and its test`) first
restores `training/scripts/train_unsloth_full.py`, launched with the container's bare `python`.
The sayso-trainer container venv is untouched. The trainer venv can stay, or be deleted with
`docker exec sayso-trainer rm -rf /workspace/host/lfm/venvs/sayso-trainer`.

## Later work (separate plans)

- TRL native prompt/completion format, with a token-parity check.
- Packing or padding-free on LFM2, once a `seq_idx` conv reset is verified on ROCm.
- DPOTrainer / GRPOTrainer behind the same contract. `--method` (default `sft`) is the slot for them.
- `trl sft --config` YAML recipes.
