# Plan: counterfactual training pairs in the generator

Date: 2026-10-21. Branch `ajax/generator-issue`, recipe `full_sft_v8.yaml`.

## Goal

Teach SaySo to decide from the Home Assistant context it is given instead of
memorizing commands. Train on groups of rows that share one command and differ
in exactly one fact the model can see:

| User says | Home Assistant context | Correct behavior |
|---|---|---|
| "Turn on the TV" | One controllable TV | Turn on the TV |
| "Turn on the TV" | TV sensor only | No action |
| "Turn on the TV" | Two matching TVs | Ask which TV |
| "Turn on the TV" | No TV | No action |
| "Turn on the TV" | TV and a "TV Active App" sensor | Turn on the TV |

**The key test:** on households and entity combinations it has never seen, does
SaySo give a different correct answer to the same command when only the context
changes?

## Current state

- `generators/grounding.py` already has counterfactual families: media
  presence, ambiguity, domain, supported action, alias, and named device. They
  come from a fixed, hand-written catalogue with one site per area and small
  name lists (`_SITE_SCREENS`, `_SITE_LAMPS`, and others). That is a few hundred
  variants serving `grounding_rate: 0.08` of a 40k corpus, and they ride on
  carrier families (`ordinary`, `absence`, `unsupported`, `clarify`).
- `scripts/split_dataset.py` keys on `template_family|phrasing|seed`. When those
  are missing it falls back to a hash of the user utterance. Variants of one
  scenario can therefore land on both sides of the split.
- `generators/validation.py` excludes eval prompts by exact normalized match
  only (`check_quality_eval_overlap`).

## Step 1: diagnose the v8 clarify shortfall

Full 40k runs leave about 34 `clarify` slots that never produce a row. The
count is the same at 20x and 30x attempts. The extra attempts all end as
`duplicate_semantic_id` or `exact_duplicate_utterance`. A clarify-only run of
2,700 slots fills every slot, so the problem appears only in the full mix.

Main suspect: `clarify` is a grounding carrier. This branch added a "surplus"
grounding top-up in `pipeline.py`, so tail clarify slots are pushed onto the
fixed grounding catalogue, which is already used up at this scale.

Diagnosis: `/tmp/deadslots/drive.py` runs the v8 mix at 20k on the VM. For every
clarify slot that never produces a row, it logs the capability, operation, home
size, real-home flag, grounding flag, grounding variant and rejection reasons.

**Result (2026-10-21).**
- At 20k, no dead slots. At 40k, 41 dead clarify slots, all lights or
  media_players turn on or off.
- 99% of their attempts carried a forced grounding variant from the fixed
  catalogue (`ground_light_s08_pair`, `ground_light_s09_pair`, and so on).
- Each variant has a fixed home and command, so after 3 uses every retry is a
  duplicate: about 46k rejections.
- Confirmed cause: once the finite catalogue is exhausted, tail grounding demand
  is forced onto the clarify carriers. The surplus top-up and v8's smaller
  `ordinary` share made that worse.

## Step 2: fix and harden

- **Root-cause fix:** depends on the diagnosis. If the catalogue is the cause,
  Step 3 replaces it.
- **Fail fast:** `_pick_family_slot` currently uses `rng.choice(open_slots or
  needed)`, so it retries closed slots until the attempt budget runs out. Make it
  stop and report the dead slots by capability, operation and rejection reason.
- **Reassign dead slots:** a slot closed after `SLOT_MAX_FAILURES` gets a new
  capability and operation that is still yielding in the same family, so one
  infeasible slot cannot fail the corpus.
- Put `max_attempts_multiplier` back to 20.

## Step 3: generated counterfactual groups

### Grouping (`generators/grounding.py`)

Replace the fixed catalogue with generated groups. A group is one base command
and one base household (from `homes.generate_home`), plus 2–5 variants. Each
variant applies exactly one mutation to the household:

| Axis | Mutation | Expected behavior |
|---|---|---|
| presence | remove the target | no action (absence) |
| domain | replace the target with a same-named sensor or binary_sensor | no action |
| count | add a second matching entity in the same area | clarify |
| area | move the target to another area (satellite-area rules apply) | act, clarify, or no action per area rules |
| feature | remove the supported operation (for example volume) | refusal |
| distractor | add a same-prefix sensor ("TV Active App") | act on the controllable entity |

Every variant keeps the command text identical, and the mutation is the only
difference that reaches the model's context. Each row is stamped with
`contrast_group`, `contrast_axis` and `contrast_variant`. The existing families
become instances of these axes.

### Labels (`generators/row_generation.py`)

- Commands come from the existing utterance and template machinery, so they
  vary, and households come from the generated pool.
- The label is never written per variant. It comes from the mutated scenario
  through `gold.py`, checked against the exposed tool schema
  (`production_catalog`).
- A new `contrast` family is planned like the others.

### Allocation

- `contrast` is 20% of rows.
- The existing grounding rows (8%) and discrimination rows (1.5%) count toward
  it, so the net new share is about 10%. It comes proportionally from `ordinary`,
  `status`, `follow_up` and `settings`.

## Step 4: split by scenario group (`scripts/split_dataset.py`)

- Assign whole groups to train, val or test before any paraphrasing. Every row
  carries a group key: `contrast_group`, or for other families the home plus the
  base semantic target.
- Hold out entire households and entity combinations:
  - reserve a slice of the name, area and device-noun pools that only test uses;
  - the split asserts that no household signature appears on both sides.

## Step 5: validation (`generators/validation.py`)

- **Conflicting labels:** reject any two rows with the same normalized command
  and the same model-visible context hash but different labels.
- **Unsupported tool calls:** every call must name an offered tool, use
  schema-valid arguments, and target an entity that supports the operation.
  This extends `_entity_can_run`.
- **Train/eval overlap:** add a fuzzy check on top of the exact match: token-set
  Jaccard of at least 0.8 against eval prompts, so "Turn on the Den TV" (4/5)
  matches the eval case "Turn on the TV".
  - **As built (2026-10-21):** the name-masked template check was dropped. Once
    names are masked, generic commands like "turn off the {X}" match some eval
    template, which banned ordinary phrasing outright.
  - **Real-home rows get the exact check only (user decision, option A).** The
    nonstandard evals reuse the real home's device names ("TV"), so a fuzzy check
    would block real-home TV training. The trade-off: the nonstandard TV cases
    partly measure phrasing the model was trained on, for the user's own TV.
  - **Synthetic homes and area scenarios never use an eval entity name**, so
    synthetic rows can't teach an eval case's answer.

## Step 6: measure

- Build a held-out contrast set from test-split groups only: unseen households,
  unseen entity combinations.
- Report per-variant accuracy and group consistency (every variant in the group
  correct). Group consistency is the key test.
- Run it as a diagnostic tag beside `nonstandard`. Promotion and smoke gates
  stay unchanged until the metric has a baseline.

## Slices (at most 4 files each, verified one at a time)

| # | Slice | Files |
|---|---|---|
| 1 | Diagnosis | none (VM run) |
| 2 | Dead-slot fail-fast and reassignment, plus the root-cause fix | `pipeline.py`, plus the cause file, tests |
| 3 | Mutation operators and group builder | `grounding.py`, tests |
| 4 | `contrast` family rows and planning | `row_generation.py`, `planning.py`, tests |
| 5 | Validation: conflicts, unsupported calls, fuzzy overlap | `validation.py`, `pipeline.py` hook, tests |
| 6 | Group and household split | `scripts/split_dataset.py`, metadata stamp, tests |
| 7 | 20% allocation in v8, generate on the VM, then `validate` and canary | `full_sft_v8.yaml` |
| 8 | Held-out contrast metric | eval script, tests |

## Open decisions

- Whether `grounding_rate` and `discrimination_rate` become sub-shares of
  `contrast` or stay separate gates. Default: they count toward the 20%.
- The holdout name and area pool fraction. Default: 15% of each pool, test only.
