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

## Delegation

All subagent and delegate work goes through the Ajax Model Router: call the
`model-router` skill, then dispatch via acpx (`scripts/run-delegate` from the
Ajax repo). Never spawn native harness subagents (Cursor Task, best-of-n,
Claude/Codex/Pi task children, or pstack explorers) for any reason, including
missing `acpx`. Never use Composer 2.5 Fast (`composer-2.5-fast` or any Fast
Composer variant) as a native Task or subagent model. Missing `acpx` is stop,
not a license to Task or parent-local writes.

Always use `model-router` for implementation writes to repository files. The
orchestrator writes plans when required, emits one `EXECUTION` decision, and
reviews delegate work. It does not explore the tree, implement, commit, push, or
open pull requests. Do not duplicate model rankings or exact model IDs in this
file. Live-host operations (running benchmarks, deploying to the GPU host) are
not repository writes and do not need delegation.

If the user explicitly approved bypassing delegation for this request, the
active agent may implement, commit, push, and open pull requests in-process.
That approval is per-request; it does not change the default.

Delegates must not merge, rebase, force-push, or switch branches unless the user
explicitly authorizes that behavior.

Every delegated task must be bounded by scope, acceptance criteria,
verification, and stop conditions. The active agent must inspect the actual
delta, confirm scope, and independently accept or reject the result. A delegate
report is evidence, not approval.

Harness-specific workflows are optional. They cannot override repository
requirements or become dependencies for other harnesses.

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
