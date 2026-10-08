# AGENTS.md

Shared repository contract for coding agents working in Ajax Optimization. Keep
this file concise, durable, and independent of any one harness. The rules below
mirror the Ajax repository's `AGENTS.md` where they apply outside that codebase.

## Scope and instruction priority

Follow instructions in this order:

1. Explicit user instruction.
2. This `AGENTS.md`.
3. `README.md`, then `docs/RESULTS.md` for measured results and rejected ideas.
4. Existing source, patches, and scripts.
5. Generated summaries or prior plans.

When instructions conflict, preserve the safest behavior and identify the
conflict. The active agent remains responsible for investigation, engineering
decisions, review, and verification, including delegated work.

Before editing, inspect the relevant files. Treat source, patches, and scripts
as authoritative over generated summaries.

## RTK

When RTK and its guidance are available through the repository, active harness,
or local environment, read that guidance and use RTK for the shell commands it
covers. Do not assume a Codex-only location or invent missing RTK behavior.

RTK must not be required for remote agents or environments where it is
unavailable. No local-machine-only file may be required for repository
correctness or remote execution.

## Repository scope

This repo holds the llama.cpp serving configuration, Vulkan kernel patches, and
benchmarks for Qwen3.8-27B on a single RX 7900 XTX, plus SaySo training and wake
tooling in `sayso-training/`. The serving goal is maximum decode tok/s inside a
fixed layout.

Serving-only hard constraints unless the user says otherwise:

- Keep 4 slots sharing one 128k unified KV pool (`--parallel -1`, `--kv-unified`,
  `-c 131072`). Propose layout changes; never apply them.
- GPU-side changes only: no CPU or RAM offload (layers, KV, experts), and do not
  add or optimize host-side work. If a GPU bottleneck traces to CPU-bound host
  code, name it and stop.
- Apply a measured tok/s win directly (back up, restart through the GPU lock,
  verify), and report what it costs. Do not ask the user to arbitrate secondary
  trade-offs.

## SaySo training

- SaySo model training design lives in
  `sayso-training/docs/SAYSO_LFM_TRAINING_PLAN.md`; wake-word training has its
  own `sayso-training/docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md`. The model target
  is `LFM2.5-230M-Base` with schema-conditioned function calling.
  `ALLOWED_HASS_TOOLS` validates the pinned training contract only — it does
  not define runtime support.
- Do not train on ChatML `<tool_call>` labels or on eval case IDs/utterances
  from `sayso-training/evals/cases/`.
- Do not expand corpora past the reviewed `sayso-training/evals/cases/`
  and training gold sets.
- Training and wake tooling runbooks are summarized in
  `sayso-training/README.md`.

## GPU host safety

- Every GPU job runs inside a lock turn: `gpu run <label> -- <cmd>`. Prod
  `llama-server` is stopped for the turn and restarts on its own afterwards.
- Never run a test server alongside prod. Put `docker rm -f` for test containers
  in a `trap`.
- A rejected or interrupted `ssh` command keeps running on the host. Run
  `docker ps -a` and check for stray `gpu run` jobs before starting anything.
- Deploy scripts and compose changes by writing a temp file and `mv` (atomic
  rename). Back up the previous file first. Never overwrite a script in place
  while a job that runs it is active.
- Benchmark from real workload shapes (about 30k depth, two concurrent streams),
  alternate A/B images, and quote medians; sampled runs swing +/-10%.

## Universal safety

- Make the smallest safe change that satisfies the request and preserve existing
  behavior unless the task explicitly changes it.
- Do not weaken, delete, skip, or rewrite tests or benchmark assertions merely to
  make a change pass. Fix implementation failures rather than weakening checks.
- Do not claim validation passed unless the command actually ran and passed.
  Never hide failed commands; report failures and skipped checks with reasons.
- Do not add generated code, large snapshots, or lockstep rewrites unless the
  task requires them.
- Update `README.md` and `docs/RESULTS.md` when behavior, commands, results, or
  workflows change.
- Never commit secrets. Reference them by environment variable name only.
- Do not force-push, and do not add `Co-Authored-By` or tool-attribution lines
  to commits or pull requests.

## Simplicity and reuse

- You are not going to need it: do not add features, abstractions, configuration,
  or dependencies for hypothetical future needs. Implement only what the current
  task requires.
- Prefer reusing existing code over writing a new equivalent. Search before you
  write.

## Delegation

The local agent does the work; the frontier agent orchestrates. All
exploration, implementation, testing, diagnosis, and reporting go through the
Ajax Model Router: call the `model-router` skill, which emits one `EXECUTION`
decision (agent, model, risk, scope, verify, fallback). The selected local
delegate runs the full loop inside that scope: explore, implement, test,
diagnose failures, and report. The frontier agent reviews the actual delta and
report and accepts or rejects it. It does not explore the tree, implement,
test, diagnose, commit, push, or open pull requests itself. A delegate report
is evidence, not approval. Live-host operations (running benchmarks, deploying
to the GPU host) are not repository writes and do not need delegation.

Never spawn native harness subagents (Cursor Task, best-of-n, Claude/Codex/Pi
task children) for any work. A delegate must run in-process. When the delegate
fails, re-route through the router; do not take over. Do not duplicate model
rankings or exact model IDs in this file.

Only an explicit user approval to bypass delegation for this request lets the
frontier agent run the loop in-process (explore, implement, test, diagnose,
report, commit, push, open pull requests). That approval is per-request; it
does not change the default, and silence or a delegate failure is not approval.

When the user asks to create a PR, the selected delegate runs the repository's
local verification gate (the checks in this file's 'Verification and
reporting' section), commits, pushes, and opens the PR with `gh pr create`; the
orchestrator reports the PR URL after reviewing the delta. After an explicit
bypass, the frontier agent does that same PR path in-process. Delegates must not
merge, rebase, force-push, or switch branches unless the user explicitly
authorizes that behavior.

Every delegated task must be bounded by scope, acceptance criteria,
verification, and stop conditions. The active agent must inspect the actual
delta, confirm scope, and independently accept or reject the result.

## No code comments

Code must never contain comments. Do not add `#` comments or docstrings to
`.py` files, and do not add `#` comments to `.sh` files — new or existing.
Shebang lines on line 1 are the only allowed `#` line. If a change seems to
need a comment, restructure the code or name it so the comment is unnecessary.

## Verification and reporting

Evidence that a change works is required. Prefer focused verification first,
then the strongest practical broader check for the risk: the FA correctness
harness (`bench/fa_correctness.sh`), the microbench (`bench/fa_perf.sh`), and an
end-to-end SOLO/PAIR A/B (`bench/fapack_ab.sh`).

If validation cannot run because of missing tools, environment limits, time, or
unrelated failures, report the exact command and result. Final reports must say
what changed, list verification and its results, disclose failed or skipped
commands, and state remaining risks or follow-up work. Do not claim the
repository is clean unless status was checked.

## Stop conditions

Stop and request direction before an unapproved change that would:

- delete user data or perform another materially destructive action;
- change the slot or KV layout, or add CPU/RAM offload;
- expose a service publicly or change authentication/security assumptions;
- reset or reboot the GPU host or stop training jobs running on it;
- perform a large rewrite not explicitly requested.

Do not stop for routine, bounded work unless the user requested an approval
gate.

## Maintaining this contract

Keep one root `AGENTS.md`. Retain here only rules needed for nearly every task;
put measured results in `docs/RESULTS.md` and runbooks next to the scripts they
describe. Do not place shared requirements only in harness-specific
configuration.
