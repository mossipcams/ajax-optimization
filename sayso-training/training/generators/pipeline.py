
from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections import ChainMap, Counter, defaultdict
from pathlib import Path
from typing import Any

from generators.capability_registry import DEFAULT_NEGATIVE_RATE, registry_summary
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.grounding import (
    GROUNDING_FAMILY_SLACK,
    generated_carrier_variant,
    grounding_capacity,
    grounding_capable_slot,
    pick_carrier_variant,
    required_training_variants,
    slot_can_carry_grounding,
    training_variants,
)
from generators.labels import render_example
from generators.pairs import PAIR_SLOT_BIAS, withheld_pair_targets, withheld_sibling
from generators.paraphrase import load_paraphraser
from generators.coverage import bare_tool_name, row_calls
from generators.planning import (
    CORE_TOOL_NAMES,
    REAL_HOME_EXCLUDED_FAMILIES,
    REAL_HOME_REFUSAL_FAMILIES,
    REAL_HOME_REFUSAL_SHARE_CAP,
    AllocationPlan,
    FamilyTracker,
    GenerationSlot,
    build_plan,
    datetime_slot,
    discrimination_capable_slot,
)
from generators.tools import namespaced_tool_name
from generators.rates import (
    discrimination_available_share,
    enforce_rate_gate,
    grounding_available_share,
)
from generators.real_home import derive_entity_cap, load_real_home
from generators.row_generation import casing_kind, generate_row
from generators.sampling import QuotaTracker
from generators.validation import (
    check_fuzzy_eval_overlap,
    check_label_conflicts,
    check_quality_eval_overlap,
    check_unsupported_calls,
)
from generators.stats import empty_stats, finalize_stats, record_accept, record_reject
from generators.validation import validate_row, validate_serialized_row, validate_spec


REAL_HOME_RETRY_LIMIT = 4

_REAL_HOME_FALLBACK_REASONS = frozenset(
    {"family_mismatch", "duplicate_semantic_id", "exact_duplicate_utterance"}
)


def _discrimination_recipe_slot(
    plan: AllocationPlan, tracker: FamilyTracker, rng: random.Random, closed: set[int]
) -> dict[str, Any] | None:
    from generators.planning import SURPLUS_OPERATIONS

    candidates = [
        slot for slot in plan.slots
        if slot.family != "area" and not slot.grounding and slot.index not in closed
        and (slot.capability, slot.operation) in SURPLUS_OPERATIONS
        and _slot_still_needed(slot, tracker, plan)
    ]
    return _slot_to_dict(rng.choice(candidates)) if candidates else None


def _recipe_required_operations(config: GeneratorConfig) -> set[tuple[int, str, str]]:
    if config.count < config.recipe_count or (config.min_positive_per_tool <= 1 and config.min_positive_per_operation <= 1):
        return set()
    from generators.capability_registry import CAPABILITIES, covered_tool_names, trainable_operations
    from generators.coverage import expected_tool

    covered = covered_tool_names() | {"__script__"}
    return {
        (cap.tier, name, op.name)
        for name, cap in CAPABILITIES.items()
        for op in trainable_operations(cap)
        if expected_tool(name, op.name) in covered
    }


def _load_excluded_prompts(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    excluded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            for msg in row.get("messages", []):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    excluded.add(msg["content"].casefold())
        except json.JSONDecodeError:
            continue
    return excluded


def _slot_still_needed(slot: GenerationSlot, tracker: FamilyTracker, plan: AllocationPlan) -> bool:
    if slot.family == "area" and slot.area_scenario:
        need = plan.area_required.get(slot.area_scenario, 0)
        return tracker.accepted_area.get(slot.area_scenario, 0) < need
    need = plan.requested.get(slot.family, 0)
    return tracker.accepted_family.get(slot.family, 0) < need


SLOT_MAX_FAILURES = 50

_DUPLICATE_REASONS = frozenset({"duplicate_semantic_id", "exact_duplicate_utterance"})


def _dead_slot_report(slots: list[GenerationSlot], reasons: dict[int, Counter[str]]) -> str:
    return "; ".join(
        f"{slot.family}/{slot.area_scenario or slot.capability}/{slot.operation}"
        f" #{slot.index}: {dict(reasons.get(slot.index, Counter()).most_common(3))}"
        for slot in slots
    )


def _pick_family_slot(
    plan: AllocationPlan,
    tracker: FamilyTracker,
    rng: random.Random,
    closed: set[int] = frozenset(),
    dead: dict[int, Counter[str]] | None = None,
) -> GenerationSlot:
    needed = [slot for slot in plan.slots if _slot_still_needed(slot, tracker, plan)]
    if not needed:
        raise RuntimeError("family tracker has no remaining shortfall slots")
    open_slots = [slot for slot in needed if slot.index not in closed]
    if open_slots:
        return rng.choice(open_slots)
    # Area slots cannot be reassigned, and each retry draws a fresh home, so a pending area slot
    # keeps being retried. Only when no such slot is left is a dead slot a real stall.
    retriable = [slot for slot in needed if slot.area_scenario]
    stuck = [slot for slot in needed if dead and slot.index in dead]
    if stuck and not retriable:
        raise RuntimeError(f"dead generation slots: {_dead_slot_report(stuck, dead)}")
    return rng.choice(retriable or needed)


def _reassign_slot(
    slot: GenerationSlot,
    yielding: Counter[tuple[str, str]],
    tried: set[tuple[str, str]],
    rng: random.Random,
) -> bool:
    from generators.capability_registry import CAPABILITIES

    if slot.area_scenario:
        return False
    tried.add((slot.capability, slot.operation))
    candidates = sorted(pair for pair in yielding if pair not in tried and pair[0] in CAPABILITIES)
    if not candidates:
        return False
    slot.capability, slot.operation = rng.choice(candidates)
    slot.tier = CAPABILITIES[slot.capability].tier
    return True


def _datetime_slot_dict(index: int) -> dict[str, Any]:
    return _slot_to_dict(datetime_slot(index))


def _slot_to_dict(slot: GenerationSlot) -> dict[str, Any]:
    return {
        "index": slot.index,
        "family": slot.family,
        "robustness": slot.robustness,
        "capability": slot.capability,
        "operation": slot.operation,
        "tier": slot.tier,
        "home_size": slot.home_size,
        "area_scenario": slot.area_scenario,
        "grounding": slot.grounding,
    }


def _feasible_area_distribution(config: GeneratorConfig) -> Path | None:
    path = config.area_distribution_path
    if path is None or not path.is_file():
        return None
    from generators.scenarios.area import assert_required_counts_feasible, load_distribution

    try:
        assert_required_counts_feasible(load_distribution(path), config.count)
    except ValueError:
        return None
    return path


def _effective_grounding_rate(config: GeneratorConfig) -> float:
    if config.real_home_path and config.real_home_rate and not config.recipe_path:
        return 0.0
    return config.grounding_rate


_HYGIENE_CORE_TOOLS = [namespaced_tool_name(name) for name in sorted(CORE_TOOL_NAMES)]
_HYGIENE_DATETIME_TOOL = namespaced_tool_name("GetDateTime")


def enforce_hygiene(
    rows: list[dict[str, Any]], count: int, *, require_datetime: bool = True
) -> dict[str, Any]:
    import importlib.util

    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    limits = yaml.safe_load(
        (repo_root / "training" / "configs" / "preflight.yaml").read_text(encoding="utf-8")
    )["limits"]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        path = Path(handle.name)
    try:
        spec = importlib.util.spec_from_file_location("sayso_preflight", repo_root / "scripts" / "preflight.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        report, errors = module.inspect_dataset(
            path,
            {
                "limits": {
                    "max_duplicate_rate": limits["max_duplicate_rate"],
                    "max_conflict_rate": limits["max_conflict_rate"],
                },
                "requirements": {
                    "minimum_rows": 0,
                    "required_positive_tools": (
                        [tool for tool in _HYGIENE_CORE_TOOLS if tool != _HYGIENE_DATETIME_TOOL]
                        if count >= 100
                        else []
                    )
                    + ([_HYGIENE_DATETIME_TOOL] if require_datetime else []),
                },
            },
            {},
        )
    finally:
        path.unlink(missing_ok=True)
    if errors:
        raise RuntimeError(f"dataset hygiene gate failed: {errors[:10]}")
    return {
        "invalid_rows": report["invalid_rows"],
        "invalid_calls": report["invalid_calls"],
        "duplicate_rate": round(report["duplicate_rate"], 5),
        "conflict_rate": round(report["conflict_rate"], 5),
        "contaminated": report["contaminated"],
    }


def run_generation(config: GeneratorConfig) -> dict[str, Any]:
    result = _run_generation(config)
    result["stats"]["hygiene"] = enforce_hygiene(
        result["rows"], config.count, require_datetime=config.get_datetime_positive_min > 0
    )
    area_path = result.get("area_distribution_path")
    if area_path:
        from generators.scenarios import area as area_scenarios

        plan = area_scenarios.load_distribution(area_path)
        result["stats"]["area_distribution"] = area_scenarios.validate_distribution(
            result["rows"], plan, config.count
        )
    from generators.manifest import build_manifest

    result["manifest"] = build_manifest(
        result["rows"],
        config_dict=config.to_dict(),
        recipe_path=config.recipe_path or Path("inline"),
        stats=result["stats"],
        allocation_summary=result["stats"].get("quota", {}),
        tokenizer_model=config.tokenizer_model,
        rejections=result["stats"].get("rejection_reasons", {}),
        token_lengths=[],
    )
    return result


class _StagedDuplicateTracker(DuplicateTracker):
    """One attempt's view of the shared tracker. Reads see both; writes stay here until commit()."""

    def __init__(self, shared: DuplicateTracker) -> None:
        super().__init__(near_limit=shared.near_limit)
        self._shared = shared
        self._pair_counts = ChainMap({}, shared._pair_counts)
        self._semantic_counts = ChainMap({}, shared._semantic_counts)
        self._specs: list[dict[str, Any]] = []

    def record(self, spec: dict[str, Any]) -> None:
        super().record(spec)
        self._specs.append(spec)

    def commit(self) -> None:
        # A row the row-level validators reject must not reserve its utterance or semantic id.
        for spec in self._specs:
            self._shared.record(spec)


def _row_rejection(row: dict[str, Any]) -> str | None:
    """Reject reason from the row-level validators, or None when the row may be accepted."""
    # Only trained turns are labels; a correction's untrained first try may name a missing entity on purpose.
    trained = {
        **row,
        "messages": [
            message for message in row["messages"]
            if message["role"] != "assistant" or message.get("train_on_turn")
        ],
    }
    if check_unsupported_calls(trained) is not None:
        return "unsupported_call"
    system = next(
        (str(m.get("content") or "") for m in row["messages"] if m["role"] == "system"), ""
    )
    # The real home shares device names with the nonstandard evals (e.g. 'TV'), so its rows get only the
    # exact normalized match; the fuzzy Jaccard check would reject ordinary real-home TV commands.
    real_home = bool(row.get("metadata", {}).get("real_home"))
    for message in row["messages"]:
        if message["role"] != "user":
            continue
        utterance = str(message.get("content") or "")
        if check_quality_eval_overlap(utterance) or (
            not real_home and check_fuzzy_eval_overlap(utterance, context=system)
        ):
            return "fuzzy_eval_overlap"
    return None


def _contrast_grounding_share(plan: AllocationPlan | None, count: int) -> float:
    # Contrast rows are grounding counterfactuals, so the planned contrast slots add to the reachable rate.
    if plan is None:
        return 0.0
    return plan.requested.get("contrast", 0) / max(count, 1)


def _withheld_pair_row(
    spec: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    dup_tracker: DuplicateTracker,
    excluded: set[str],
) -> dict[str, Any] | None:
    rate = config.pair_rates.get("withheld_tool", 0.0)
    if rate <= 0.0 or rng.random() >= rate:
        return None
    sibling_spec = withheld_sibling(spec, rng)
    if sibling_spec is None or validate_spec(sibling_spec) is not None:
        return None
    utterance = sibling_spec["utterance"]
    if check_quality_eval_overlap(utterance) or utterance.casefold() in excluded:
        return None
    if validate_row(sibling_spec, token_budget=config.token_budget) is not None:
        return None
    if dup_tracker.would_reject(sibling_spec):
        return None
    try:
        sibling_row = render_example(sibling_spec)
    except ValueError:
        return None
    if validate_serialized_row(
        sibling_row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    ) is not None:
        return None
    sibling_row["metadata"]["pair_kind"] = "withheld_tool"
    dup_tracker.record(sibling_spec)
    return sibling_row


def _run_generation(config: GeneratorConfig) -> dict[str, Any]:
    rng = random.Random(config.seed)
    grounding_rate = _effective_grounding_rate(config)
    grounding_target = int(round(config.count * grounding_rate))
    capacity = grounding_capacity(config)
    if grounding_target > capacity:
        raise ValueError(
            f"grounding_rate {grounding_rate} on {config.count} rows asks for "
            f"{grounding_target} grounding rows, but the catalogue can produce at most "
            f"{capacity} ({len(training_variants())} scenarios x near_duplicate_limit "
            f"{config.near_duplicate_limit}). Add sites in generators.grounding._sites "
            "or lower the rate."
        )
    datetime_target = config.scaled_floor(config.get_datetime_positive_min)
    datetime_accepted = 0
    plan = None
    tracker = None
    quota = None
    area_distribution_path = _feasible_area_distribution(config)
    if config.recipe_path:
        if not config.allocations:
            raise ValueError("recipe YAML must specify allocations")
        plan = build_plan(
            config.count,
            config.seed,
            config.allocations,
            area_distribution_path=area_distribution_path,
            near_duplicate_limit=config.near_duplicate_limit,
            grounding_variants=len(training_variants()),
            grounding_rate=grounding_rate,
            datetime_required=datetime_target,
            real_home_path=config.real_home_path,
            real_home_rate=config.real_home_rate,
            min_positive_per_tool=config.scaled_floor(config.min_positive_per_tool),
            min_positive_per_operation=config.scaled_floor(config.min_positive_per_operation),
        )
        tracker = FamilyTracker(plan)
    elif config.allocations:
        plan = build_plan(
            config.count,
            config.seed,
            config.allocations,
            area_distribution_path=area_distribution_path,
            near_duplicate_limit=config.near_duplicate_limit,
            grounding_variants=len(training_variants()),
            grounding_rate=grounding_rate,
            datetime_required=datetime_target,
            real_home_path=config.real_home_path,
            real_home_rate=config.real_home_rate,
        )
        tracker = FamilyTracker(plan)
    else:
        quota = QuotaTracker(
            config.count,
            config.seed,
            config.tier_proportions,
            negative_rate=DEFAULT_NEGATIVE_RATE,
        )


    excluded = _load_excluded_prompts(config.exclude_prompts_path)
    dup_tracker = DuplicateTracker(near_limit=config.near_duplicate_limit)
    real_home_targets: Counter[str] = Counter()
    real_home_usage: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    real_home_names: frozenset[str] = frozenset()
    real_home_entity_cap = config.real_home_entity_cap
    if config.real_home_path:
        real_entities = load_real_home(config.real_home_path, split="train")["entities"]
        real_home_names = frozenset(entity["name"] for entity in real_entities)
        if not real_home_entity_cap:
            real_home_entity_cap = derive_entity_cap(
                config.count, config.real_home_rate, len(real_entities)
            )
    stats = empty_stats()
    accepted: list[dict[str, Any]] = []
    semantic_ids: set[str] = set()
    accepted_prompts: set[str] = set()
    real_home_rows = 0
    real_home_refusal_rows = 0
    grounding_rows: Counter[str] = Counter()
    required_grounding = required_training_variants()
    groundable_pairs = {(variant["capability"], variant["operation"]) for variant in training_variants()}
    grounding_missing = (
        {variant["family"]: variant for variant in required_grounding}
        if grounding_rate > 0
        and config.count * grounding_rate
        >= GROUNDING_FAMILY_SLACK * len(required_grounding)
        else {}
    )
    discrimination_target = int(round(config.count * config.discrimination_rate))
    discrimination_rows = 0
    real_home_target = int(round(config.count * config.real_home_rate))
    balance_real_home = bool(
        config.real_home_path and config.real_home_rate and config.real_home_entity_cap == 0
    )
    attempts = 0
    max_attempts = config.max_attempts()

    load_paraphraser(config.paraphrase_enabled)
    stt_target = int(round(config.count * config.stt_noise_rate))
    stt_log_target = int(round(config.count * config.stt_log_rate))

    def accepted_total() -> int:
        return len(accepted)

    def complete() -> bool:
        if tracker is not None:
            return len(accepted) >= config.count
        return (
            datetime_accepted >= datetime_target
            and quota.accepted_total() + datetime_accepted >= config.count
        )

    unused_grounding_slots = (
        {slot.index for slot in plan.slots if slot.grounding} if plan is not None else set()
    )
    closed_slots: set[int] = set()
    casing_counts: dict[str, Counter[str]] = {"call": Counter(), "no_call": Counter()}
    pair_targets = withheld_pair_targets() if config.pair_rates.get("withheld_tool") else []
    slot_failures: Counter[int] = Counter()
    slot_reasons: defaultdict[int, Counter[str]] = defaultdict(Counter)
    dead_slots: dict[int, Counter[str]] = {}
    slot_tried: defaultdict[int, set[tuple[str, str]]] = defaultdict(set)
    family_yield: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    stale_variants: defaultdict[int, set[str]] = defaultdict(set)
    while not complete() and attempts < max_attempts:
        family_slot = (
            _pick_family_slot(plan, tracker, rng, closed_slots, dead_slots)
            if tracker is not None else None
        )
        rows_remaining = max(1, config.count - accepted_total())
        datetime_deficit = datetime_target - datetime_accepted
        if datetime_deficit > 0 and rows_remaining <= datetime_deficit:
            slot = _datetime_slot_dict(attempts)
        elif family_slot is not None:
            slot = _slot_to_dict(family_slot)
            unused_grounding_slots.discard(slot["index"])
        else:
            slot = quota.next_slot()
        real_home_selected = None
        real_home_forced = False
        if balance_real_home:
            real_home_remaining = max(0, real_home_target - real_home_rows)
            slot_family = slot.get("family")
            if slot_family in REAL_HOME_EXCLUDED_FAMILIES:
                real_home_selected = False
            else:
                real_home_forced = real_home_remaining >= rows_remaining
                real_home_selected = real_home_forced or rng.random() < real_home_remaining / rows_remaining

        grounding_deficit = grounding_target - sum(grounding_rows.values())
        discrimination_deficit = discrimination_target - discrimination_rows
        carrier = quota is None and slot_can_carry_grounding(slot)
        want_grounding = False
        recipe_variant = None
        # Contrast rows can meet the rate before every catalogue family is covered, so the catalogue is
        # still offered while families are missing; past the rate, only those missing families are taken.
        if grounding_rate > 0 and (grounding_deficit > 0 or grounding_missing):
            rate_hit = grounding_deficit > 0 and rng.random() < min(1.0, grounding_deficit / rows_remaining)
            if quota is not None:
                want_grounding = bool(grounding_missing) or rate_hit
            elif carrier:
                variant = pick_carrier_variant(
                    slot, rng, grounding_missing, stale_variants[slot["index"]]
                )
                surplus = rate_hit and grounding_deficit > len(unused_grounding_slots)
                if variant is not None:
                    want_grounding = bool(slot.get("grounding") or rng.random() < 0.5 or surplus)
                elif grounding_deficit > 0 and (slot.get("grounding") or surplus):
                    variant = generated_carrier_variant(slot["family"], attempts, config.seed)
                    want_grounding = variant is not None
                if want_grounding:
                    slot = {
                        **slot,
                        "capability": variant["capability"],
                        "operation": variant["operation"],
                    }
                    recipe_variant = [variant]
        discrimination_index = None
        want_discrimination = (
            config.discrimination_rate > 0
            and discrimination_deficit > 0
            and not want_grounding
            and rng.random() < max(0.5, min(1.0, 2.0 * discrimination_deficit / rows_remaining))
        )
        if want_grounding and quota is not None:
            forced = grounding_capable_slot(quota, rng)
            if forced is not None:
                slot = forced
            elif (slot["capability"], slot["operation"]) not in groundable_pairs:
                # e.g. the datetime slot: no grounding variant can serve it, so it would miss every attempt.
                want_grounding = False
        elif want_discrimination:
            forced = (
                discrimination_capable_slot(quota, rng)
                if quota is not None
                else _discrimination_recipe_slot(plan, tracker, rng, closed_slots)
            )
            if forced is not None:
                slot = forced
                slot["home_size"] = max(slot["home_size"], 64)
                if quota is None:
                    discrimination_index = forced["index"]
        if (
            pair_targets
            and tracker is not None
            and not want_grounding
            and not want_discrimination
            and slot.get("family") in (None, "ordinary")
            and not slot.get("grounding")
            and tracker.family_deficit("unavailable") > 0
            and rng.random() < PAIR_SLOT_BIAS
        ):
            slot = {**slot, **rng.choice(pair_targets)}
        row, reason = None, None
        exclusion_cadence = (
            config.real_home_entity_cap > 0
            and attempts % 125 == 0
            and not slot.get("family")
            and slot["capability"] in {"lights", "switches", "fans", "covers"}
            and slot["operation"] in {"turn_on", "turn_off", "open", "close"}
        )
        real_home_retries = 0
        if (
            slot.get("family") in REAL_HOME_REFUSAL_FAMILIES
            and real_home_refusal_rows + 1 > REAL_HOME_REFUSAL_SHARE_CAP * (real_home_rows + 1)
        ):
            real_home_selected = False
            real_home_forced = False
        while True:
            if exclusion_cadence:
                real_sel = real_home_selected
            elif (
                real_home_retries == 1
                and real_home_forced
                and reason in _REAL_HOME_FALLBACK_REASONS
            ):
                real_sel = False
            else:
                real_sel = True if real_home_retries else real_home_selected
            row_rng = (
                random.Random(
                    (config.seed ^ (attempts * 7919) ^ (real_home_retries * 104729))
                    & 0xFFFFFFFF
                )
                if real_home_retries
                else rng
            )
            staged_dups = _StagedDuplicateTracker(dup_tracker)
            row, reason = generate_row(
                slot,
                config,
                row_rng,
                excluded=excluded,
                dup_tracker=staged_dups,
                attempt=attempts,
                retry=real_home_retries,
                stt_remaining=max(
                    0,
                    stt_target - (stats["stt_corrupted"] - stats["stt_log_corrupted"]),
                ),
                stt_log_remaining=max(0, stt_log_target - stats["stt_log_corrupted"]),
                rows_remaining=rows_remaining,
                real_home_targets=real_home_targets,
                real_home_entity_cap=real_home_entity_cap,
                real_home_names=real_home_names,
                real_home_usage=real_home_usage[(slot["capability"], slot["operation"])],
                quota=quota,
                grounding_required=want_grounding,
                discrimination_required=want_discrimination,
                real_home_selected=real_sel,
                grounding_variants=recipe_variant or (
                    list(grounding_missing.values())
                    if want_grounding and grounding_missing
                    else None
                ),
                grounding_rate=(
                    grounding_rate if quota is not None or want_grounding else 0.0
                ),
                casing_counts=casing_counts,
            )
            # Row-level rejections are retried like generator rejections, so a real-home row is not lost to them.
            if row is not None:
                reason = _row_rejection(row)
                if reason is None:
                    staged_dups.commit()
                    break
                row = None
            else:
                staged_dups.commit()
            if real_home_selected is not True and not exclusion_cadence:
                break
            real_home_retries += 1
            if real_home_retries >= REAL_HOME_RETRY_LIMIT:
                break
        attempts += 1
        plan_index = discrimination_index if discrimination_index is not None else (
            family_slot.index if family_slot is not None else None
        )
        if recipe_variant and row is None and reason in _DUPLICATE_REASONS:
            stale_variants[slot["index"]].add(recipe_variant[0]["family"])
        if plan_index is not None and slot.get("index") == plan_index:
            if row is not None:
                closed_slots.add(plan_index)
                if not want_grounding:
                    family_yield[slot["family"]][(slot["capability"], slot["operation"])] += 1
            else:
                slot_failures[plan_index] += 1
                slot_reasons[plan_index][reason or "unknown"] += 1
                if slot_failures[plan_index] >= SLOT_MAX_FAILURES:
                    planned = next(item for item in plan.slots if item.index == plan_index)
                    if _reassign_slot(planned, family_yield[planned.family], slot_tried[plan_index], rng):
                        slot_failures[plan_index] = 0
                    else:
                        closed_slots.add(plan_index)
                        if not planned.area_scenario:
                            dead_slots[plan_index] = slot_reasons[plan_index]
        if row is None:
            record_reject(stats, reason or "unknown")
            continue
        pair_spec = row.pop("_pair_spec", None)
        if config.split == "test":
            user_text = next(
                message["content"]
                for message in row["messages"]
                if message["role"] == "user"
            )
            folded = user_text.casefold()
            if folded in accepted_prompts:
                record_reject(stats, "duplicate_utterance")
                continue
            accepted_prompts.add(folded)
        sem = row.get("metadata", {}).get("semantic_id")
        if sem:
            semantic_ids.add(sem)
        if tracker is not None:
            tracker.record(slot["family"], slot.get("area_scenario"))
        elif slot.get("family") != "datetime":
            quota.record_accept(row)
        accepted.append(row)
        record_accept(stats, row)
        first_user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        casing_counts[casing_kind(bool(row_calls(row)))]["upper" if first_user[:1].isupper() else "lower"] += 1
        if any(
            bare_tool_name(call["name"]) == "GetDateTime"
            for call in row_calls(row)
        ):
            datetime_accepted += 1
        real_home_rows += bool(row["metadata"].get("real_home"))
        real_home_refusal_rows += bool(
            row["metadata"].get("real_home")
            and row["metadata"].get("family") in REAL_HOME_REFUSAL_FAMILIES
        )
        if row["metadata"].get("grounding_family"):
            family = row["metadata"]["grounding_family"]
            grounding_rows[family] += 1
            grounding_missing.pop(family, None)
        if row["metadata"].get("discrimination"):
            discrimination_rows += 1
        if pair_spec is not None and tracker is not None and len(accepted) < config.count:
            if tracker.family_deficit("unavailable") > 0:
                sibling_row = _withheld_pair_row(pair_spec, config, rng, dup_tracker, excluded)
                if sibling_row is not None:
                    row["metadata"]["contrastive_group"] = f"withheld_{pair_spec.get('candidate_id')}"
                    sibling_row["metadata"]["contrastive_group"] = row["metadata"]["contrastive_group"]
                    tracker.record("unavailable")
                    accepted.append(sibling_row)
                    record_accept(stats, sibling_row)

    if not complete():
        raise RuntimeError(
            f"failed to meet allocation: accepted {accepted_total()}/{config.count} "
            f"after {attempts} attempts; shortfall="
            f"{tracker.summary() if tracker is not None else quota.shortfall()}; "
            f"missing_grounding={sorted(grounding_missing)}; "
            f"rejections={dict(stats['rejection_reasons'])}"
        )

    conflicts = check_label_conflicts(accepted)
    if conflicts:
        shown = ", ".join(repr(conflict["command"]) for conflict in conflicts[:5])
        raise RuntimeError(
            f"{len(conflicts)} accepted command/context group(s) disagree on the label; "
            f"first commands: {shown}"
        )

    if tracker is not None:
        tracker.verify_complete()
        if datetime_accepted < datetime_target:
            raise RuntimeError(
                f"missing GetDateTime positive rows: got {datetime_accepted}, "
                f"need {datetime_target}"
            )
    else:
        if datetime_accepted < datetime_target:
            raise RuntimeError(
                f"missing GetDateTime positive rows: got {datetime_accepted}, "
                f"need {datetime_target}"
            )
        gaps = quota.shortfall()
        operation_gap = sum(gaps.get("operation", {}).values())
        if operation_gap > datetime_target:
            quota.verify_complete()
    if grounding_missing:
        raise RuntimeError(
            f"missing required grounding families: {sorted(grounding_missing)}"
        )
    report = finalize_stats(
        stats,
        semantic_ids,
        quota_summary=tracker.summary() if tracker is not None else quota.summary(),
    )
    report["requested_stt_rate"] = config.stt_noise_rate
    report["achieved_stt_rate"] = round(stats["stt_corrupted"] / max(stats["accepted"], 1), 4)
    report["requested_stt_log_rate"] = config.stt_log_rate
    report["achieved_stt_log_rate"] = round(
        stats["stt_log_corrupted"] / max(stats["accepted"], 1), 4
    )
    report["requested"] = config.count
    report["attempts"] = attempts
    report["config"] = config.to_dict()
    report["grounding"] = {
        "requested_rate": grounding_rate,
        "rows": sum(grounding_rows.values()),
        "achieved_rate": round(sum(grounding_rows.values()) / max(len(accepted), 1), 4),
        "by_family": dict(sorted(grounding_rows.items())),
    }
    report["discrimination"] = {
        "requested_rate": config.discrimination_rate,
        "rows": discrimination_rows,
        "achieved_rate": round(discrimination_rows / max(len(accepted), 1), 4),
    }
    if grounding_rate > 0:
        enforce_rate_gate(
            report["grounding"],
            "grounding",
            config,
            len(accepted),
            available_share=grounding_available_share(config)
            + _contrast_grounding_share(plan, config.count),
        )
    enforce_rate_gate(
        report["discrimination"],
        "discrimination",
        config,
        len(accepted),
        available_share=discrimination_available_share(config),
    )
    for key, requested in (("full_tool_catalog", 1.0),):
        rows = sum(1 for row in accepted if row["metadata"].get(key))
        report[key] = {
            "requested_rate": requested,
            "rows": rows,
            "achieved_rate": round(rows / max(len(accepted), 1), 4),
        }
        enforce_rate_gate(report[key], key, config, len(accepted))
    if config.real_home_path:
        report["real_home"] = {
            "path": str(config.real_home_path),
            "requested_rate": config.real_home_rate,
            "synthetic_only": config.synthetic_only,
            "rows": real_home_rows,
            "refusal_rows": real_home_refusal_rows,
            "refusal_share_cap": REAL_HOME_REFUSAL_SHARE_CAP,
            "achieved_rate": round(real_home_rows / max(len(accepted), 1), 4),
            "entity_cap": real_home_entity_cap,
            "entities_used": len(real_home_targets),
            "target_counts": dict(sorted(real_home_targets.items())),
            "most_common": real_home_targets.most_common(5),
        }
    report["registry"] = registry_summary()
    from generators.audit import audit_rows

    report["quality_audit"] = audit_rows(
        accepted,
        expected_count=config.count,
        required_operations=(
            _recipe_required_operations(config)
            if quota is None
            else {key for key, target in quota.targets["positive"].items() if target > 0}
        ),
        min_positive_per_operation=config.scaled_floor(config.min_positive_per_operation),
        min_positive_per_tool=config.scaled_floor(config.min_positive_per_tool),
        max_absence_rate=config.max_absence_rate,
        get_datetime_positive_min=config.scaled_floor(config.get_datetime_positive_min),
        allowed_operation_shortfall=(
            config.get_datetime_positive_min if quota is not None else 0
        ),
    )
    return {
        "rows": accepted,
        "stats": report,
        "area_distribution_path": area_distribution_path,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_build(config: GeneratorConfig) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="sayso_gen_"))
    try:
        result = run_generation(config)
        data = tmp / "dataset.jsonl"
        manifest = tmp / "manifest.json"
        write_jsonl(data, result["rows"])
        from generators.manifest import write_manifest as write_canonical_manifest

        write_canonical_manifest(manifest, result["manifest"])
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = config.manifest_path or config.output_path.with_suffix(".manifest.json")
        shutil.move(data, config.output_path)
        shutil.move(manifest, manifest_path)
        result.update(output_path=str(config.output_path), manifest_path=str(manifest_path))
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_manifest(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in report.items() if k != "rows"}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
