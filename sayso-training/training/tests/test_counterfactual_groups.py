from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker, _normalize
from generators.gold import target_names_from_expected
from generators.planning import (
    PRIMARY_FAMILIES,
    REAL_HOME_EXCLUDED_FAMILIES,
    AllocationPlan,
    FamilyTracker,
    GenerationSlot,
    build_plan,
)
from generators.row_generation import generate_row
from generators.tools import namespaced_tool_name
from generators import validation
from generators.validation import check_fuzzy_eval_overlap, check_quality_eval_overlap
from generators import grounding
from generators.pipeline import _run_generation, run_generation
from generators.rates import RATE_ACHIEVED_TOLERANCE
from generators.grounding import (
    CONTRAST_AXES,
    CONTRAST_GROUP_SIZE,
    GENERATED_CARRIER_AXES,
    behaviour_class,
    build_contrast_group,
    build_spec,
    contrast_gold,
    entity,
    generated_carrier_variant,
    home,
    model_visible_facts,
    mutate_count,
    mutate_distractor,
    mutate_domain,
    mutate_presence,
    seeded_contrast_group,
)


def _tv_home():
    return home(
        "tv_table",
        [entity("TV", "media_players", "Living Room", device_class="tv",
                features=("on", "off", "play", "pause", "volume", "mute")),
         entity("Floor Lamp", "lights", "Living Room")],
        sayso_entity_area="Living Room",
    )


def _turn_on_tv(household):
    expected = contrast_gold(household, "media_players", "turn_on", "TV")
    return behaviour_class(expected), target_names_from_expected(expected)


def test_tv_table_from_the_plan():
    base = _tv_home()
    snapshot = model_visible_facts(base)
    assert _turn_on_tv(base) == ("action", ["TV"])
    assert _turn_on_tv(mutate_domain(base, "TV"))[0] == "no_action"
    assert _turn_on_tv(mutate_count(base, "TV"))[0] == "clarify"
    assert _turn_on_tv(mutate_presence(base, "TV"))[0] == "no_action"
    distracted = mutate_distractor(base, "TV")
    assert "tv active app" in model_visible_facts(distracted)
    assert _turn_on_tv(distracted) == ("action", ["TV"])
    assert model_visible_facts(base) == snapshot


def _differing(left, right):
    a, b = model_visible_facts(left), model_visible_facts(right)
    return {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}


def test_variants_share_the_command_and_change_one_fact():
    for index in range(30):
        group = build_contrast_group(index, random.Random(index))
        assert 2 <= len(group) <= 5
        base = group[0]
        assert base["contrast_axis"] == "base" and base["expected_class"] == "action"
        assert len({item["utterance"] for item in group}) == 1
        assert len({item["contrast_group"] for item in group}) == 1
        assert [item["contrast_variant"] for item in group] == list(range(len(group)))
        assert len({item["contrast_axis"] for item in group}) == len(group)
        target = base["target_name"].casefold()
        for item in group[1:]:
            changed = _differing(base["home"], item["home"])
            assert len(changed) == 1, (item["contrast_axis"], changed)
            if item["contrast_axis"] in {"presence", "domain", "feature"}:
                assert changed == {target}
            if item["contrast_axis"] == "area":
                assert item["noun"].casefold() in next(iter(changed))
        for left in group:
            for right in group:
                touched = _differing(base["home"], left["home"]) | _differing(base["home"], right["home"])
                assert _differing(left["home"], right["home"]) <= touched


def test_class_comes_from_gold_on_the_mutated_household():
    expected = {"presence": "no_action", "domain": "no_action", "count": "clarify",
                "feature": "refusal", "distractor": "action"}
    for index in range(30):
        for item in build_contrast_group(index, random.Random(index)):
            spec = build_spec(item, index=index)
            assert behaviour_class(spec["expected"]) == item["expected_class"]
            if item["contrast_axis"] in expected:
                assert item["expected_class"] == expected[item["contrast_axis"]]


def test_groups_are_deterministic():
    for index in (0, 7, 123):
        assert build_contrast_group(index, random.Random(index)) == build_contrast_group(
            index, random.Random(index)
        )


def test_groups_cover_every_axis():
    seen = {
        item["contrast_axis"]
        for index in range(120)
        for item in build_contrast_group(index, random.Random(index))
    }
    assert seen == {"base", *CONTRAST_AXES}


def test_forced_axes_build_that_group():
    group = build_contrast_group(5, random.Random(5), axes=("feature", "count"))
    assert [item["contrast_axis"] for item in group] == ["base", "feature", "count"]
    assert [item["expected_class"] for item in group] == ["action", "refusal", "clarify"]


SEED = 7
SAMPLE = 200


def _slot(index):
    return {"index": index, "family": "contrast", "robustness": "ambiguity",
            "capability": "lights", "operation": "turn_on", "tier": 1, "home_size": 32}


def _generate(count=SAMPLE):
    config = GeneratorConfig(count=count, seed=SEED)
    tracker = DuplicateTracker()
    rows = []
    for index in range(count):
        row, reason = generate_row(_slot(index), config, random.Random(index),
                                   excluded=set(), dup_tracker=tracker)
        assert row is not None, (index, reason)
        rows.append(row)
    return rows


@lru_cache(maxsize=1)
def _sample():
    return tuple(_generate())


def _command(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def _trained(row):
    return json.dumps([
        (m.get("content"), [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []])
        for m in row["messages"] if m["role"] == "assistant"
    ])


def _visible(row):
    return json.dumps([row["messages"][0]["content"], sorted(t["function"]["name"] for t in row["tools"])])


def _tool_names(row):
    return {tool["function"]["name"] for tool in row["tools"]}


def test_contrast_is_planned_like_the_other_multi_variant_families():
    assert "contrast" not in PRIMARY_FAMILIES
    assert "contrast" in REAL_HOME_EXCLUDED_FAMILIES
    plan = build_plan(40, 3, {"contrast": 0.5, "ordinary": 0.5})
    contrast = sorted(slot.index for slot in plan.slots if slot.family == "contrast")
    assert contrast == list(range(20))
    assert {slot.robustness for slot in plan.slots if slot.family == "contrast"} == {"ambiguity"}


def test_rows_of_a_group_share_the_command_and_carry_gold_labels():
    groups = defaultdict(list)
    for row in _sample():
        groups[row["metadata"]["contrast_group"]].append(row)
    assert len(groups) == SAMPLE // CONTRAST_GROUP_SIZE
    for index, (group_id, rows) in enumerate(groups.items()):
        assert len({_command(row) for row in rows}) == 1
        assert [row["metadata"]["contrast_variant"] for row in rows] == list(range(CONTRAST_GROUP_SIZE))
        for row, item in zip(rows, seeded_contrast_group(index, SEED)):
            assert item["contrast_group"] == group_id
            assert row["metadata"]["family"] == "contrast"
            assert row["metadata"]["contrast_axis"] == item["contrast_axis"]
            expected = item["expected"]
            calls = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                     for m in row["messages"] for c in m.get("tool_calls") or []]
            assert calls == [(namespaced_tool_name(c["name"]), c["arguments"]) for c in expected["calls"]]
            assert row["metadata"]["no_action_reason"] == expected.get("response")
            assert {name for name, _ in calls} <= _tool_names(row)


def test_no_contrast_command_matches_an_eval_prompt():
    assert not any(check_quality_eval_overlap(_command(row)) for row in _sample())


def test_no_contrast_command_is_fuzzy_eval_overlap_across_groups():
    for index in range(120):
        group = build_contrast_group(index, random.Random(index))
        assert not check_fuzzy_eval_overlap(group[0]["utterance"]), index


def test_a_fuzzy_overlapping_phrasing_is_replaced_by_a_clean_one(monkeypatch):
    base = build_contrast_group(0, random.Random(0))[0]
    real = validation.check_fuzzy_eval_overlap
    rejected = []

    def flag_first(utterance, **kwargs):
        if not rejected:
            rejected.append(utterance)
            return True
        return real(utterance, **kwargs)

    monkeypatch.setattr(validation, "check_fuzzy_eval_overlap", flag_first)
    command = grounding.contrast_command(
        base["home"], base["target_name"], base["expected"], base["noun"], base["contrast_group"]
    )
    assert rejected and command != rejected[0]
    assert not real(command)


def test_area_variants_can_act_or_clarify():
    outcomes = {
        row["metadata"]["no_action_reason"]
        for row in _sample() if row["metadata"]["contrast_axis"] == "area"
    }
    assert None in outcomes and "clarify" in outcomes and "device_absent" in outcomes


def test_feature_variant_withholds_the_unsupported_tool():
    rows = _sample()
    base = {row["metadata"]["contrast_group"]: row for row in rows if row["metadata"]["contrast_variant"] == 0}
    feature = [row for row in rows if row["metadata"]["contrast_axis"] == "feature"]
    assert feature
    for row in feature:
        assert row["metadata"]["no_action_reason"] == "device_unsupported"
        missing = _tool_names(base[row["metadata"]["contrast_group"]]) - _tool_names(row)
        assert len(missing) == 1 and not missing & {namespaced_tool_name("HassTurnOn"), namespaced_tool_name("HassTurnOff")}


def test_no_command_and_visible_context_pair_has_two_labels():
    labels = defaultdict(set)
    for row in _sample():
        labels[(_normalize(_command(row)), _visible(row))].add(_trained(row))
    assert all(len(found) == 1 for found in labels.values())


def test_contrast_rows_are_deterministic():
    assert _generate(2 * CONTRAST_GROUP_SIZE) == list(_sample()[: 2 * CONTRAST_GROUP_SIZE])


def _clear_grounding_caches():
    grounding.carrier_variants.cache_clear()
    grounding.grounding_pairs.cache_clear()
    grounding._catalogue.cache_clear()


@pytest.fixture
def small_catalogue(monkeypatch):
    _clear_grounding_caches()
    monkeypatch.setattr(grounding, "_catalogue", lambda: tuple(grounding.canonical_variants()))
    yield
    monkeypatch.undo()
    _clear_grounding_caches()


def test_generated_carrier_variants_match_their_family():
    for family, axes in GENERATED_CARRIER_AXES.items():
        seen = [generated_carrier_variant(family, index, SEED) for index in range(12)]
        made = [item for item in seen if item is not None]
        assert len(made) >= 10, family
        assert len({item["home"]["home_id"] for item in made}) == len(made)
        for item in made:
            assert item["family"] in {f"ground_generated_{axis}" for axis in axes}
            assert behaviour_class(build_spec(item)["expected"]) == behaviour_class(item["expected"])


def test_blocked_request_slots_never_get_a_carrier_variant():
    from generators.planning import BLOCKED_REQUEST_OPERATIONS

    all_families = tuple({v["family"] for v in grounding.carrier_variants("unsupported")})
    for index, (capability, operation) in enumerate(BLOCKED_REQUEST_OPERATIONS * 4):
        slot = {"index": index, "family": "unsupported", "capability": capability,
                "operation": operation, "grounding": True}
        assert grounding.pick_carrier_variant(slot, random.Random(index), all_families) is None
    assert all(
        generated_carrier_variant("unsupported", index, SEED) is None for index in range(12)
    )
    ordinary = {"index": 0, "family": "unsupported", "capability": "lights", "operation": "turn_on"}
    for index in range(20):
        picked = grounding.pick_carrier_variant(ordinary, random.Random(index), all_families)
        assert picked is None or not grounding.is_blocked_request(
            "unsupported", picked["capability"], picked["operation"]
        )


def test_generated_carrier_variants_never_target_a_context_sensor():
    for family, _axes in GENERATED_CARRIER_AXES.items():
        for index in range(12):
            item = generated_carrier_variant(family, index, SEED)
            if item is None:
                continue
            assert not grounding._targets_context_sensor(item), (family, index)


def test_exhausted_catalogue_still_fills_clarify_and_counts_grounding(small_catalogue):
    config = GeneratorConfig(
        count=400, seed=SEED,
        allocations={"clarify": 0.6, "ordinary": 0.2, "absence": 0.1, "unsupported": 0.1},
        grounding_rate=0.15, near_duplicate_limit=3,
    )
    result = _run_generation(config)
    rows, stats = result["rows"], result["stats"]
    achieved = stats["quota"]["achieved_family"]
    requested = stats["quota"]["requested"]
    assert achieved["clarify"] == requested["clarify"]
    report = stats["grounding"]
    required = {variant["family"] for variant in grounding.required_training_variants()}
    assert required <= set(report["by_family"])
    assert report["rows"] >= RATE_ACHIEVED_TOLERANCE * round(config.count * config.grounding_rate)
    generated_clarify = [
        row for row in rows
        if row["metadata"].get("family") == "clarify"
        and row["metadata"].get("grounding_family") == "ground_generated_count"
    ]
    assert generated_clarify
    assert report["rows"] == sum(1 for row in rows if row["metadata"].get("grounding_family"))


def _infeasible_run(monkeypatch, *, pair_only):
    from generators import pipeline

    victim = {}
    real = pipeline.generate_row

    def generate(slot, *args, **kwargs):
        if slot.get("family") == "clarify" and not victim:
            victim.update(index=slot["index"], pair=(slot["capability"], slot["operation"]))
        if slot.get("index") == victim.get("index") and (
            not pair_only or (slot["capability"], slot["operation"]) == victim["pair"]
        ):
            victim["failures"] = victim.get("failures", 0) + 1
            return None, "family_mismatch"
        row, reason = real(slot, *args, **kwargs)
        if row is not None and slot.get("index") == victim.get("index"):
            victim["filled_with"] = (slot["capability"], slot["operation"])
        return row, reason

    monkeypatch.setattr(pipeline, "generate_row", generate)
    config = GeneratorConfig(count=80, seed=SEED, allocations={"clarify": 0.5, "ordinary": 0.5})
    return config, victim


def test_an_infeasible_slot_is_reassigned_to_a_yielding_pair(monkeypatch):
    from generators.pipeline import SLOT_MAX_FAILURES

    config, victim = _infeasible_run(monkeypatch, pair_only=True)
    result = run_generation(config)
    assert len(result["rows"]) == config.count
    assert victim["failures"] == SLOT_MAX_FAILURES
    assert victim["filled_with"] != victim["pair"]


def test_a_dead_slot_raises_a_readable_report_before_the_budget(monkeypatch):
    from generators.pipeline import SLOT_MAX_FAILURES

    config, victim = _infeasible_run(monkeypatch, pair_only=False)
    with pytest.raises(RuntimeError, match=r"dead generation slots: clarify/\w+/\w+ #\d+: \{'family_mismatch'"):
        run_generation(config)
    assert victim["failures"] % SLOT_MAX_FAILURES == 0
    assert victim["failures"] < config.max_attempts()


def _retry_plan():
    ordinary = GenerationSlot(0, "clarify", "ambiguity", "lights", "turn_on", 1, 8)
    area = GenerationSlot(
        1, "area", "ambiguity", "lights", "turn_off", 1, 8,
        area_scenario="exact_name_outside_satellite_area",
    )
    plan = AllocationPlan(
        count=2,
        requested={"clarify": 1, "area": 1},
        slots=[ordinary, area],
        area_required={"exact_name_outside_satellite_area": 1},
    )
    return plan, FamilyTracker(plan), ordinary, area


def test_a_closed_area_slot_keeps_being_retried_while_an_ordinary_slot_is_dead():
    from collections import Counter

    from generators.pipeline import _pick_family_slot

    plan, tracker, ordinary, area = _retry_plan()
    dead = {ordinary.index: Counter({"family_mismatch": 50})}
    picked = _pick_family_slot(plan, tracker, random.Random(0), closed={0, 1}, dead=dead)
    assert picked is area


def test_a_dead_ordinary_slot_raises_once_no_area_slot_is_pending():
    from collections import Counter

    from generators.pipeline import _pick_family_slot

    plan, tracker, ordinary, area = _retry_plan()
    dead = {ordinary.index: Counter({"family_mismatch": 50})}
    tracker.accepted_area["exact_name_outside_satellite_area"] = 1
    with pytest.raises(RuntimeError, match=r"dead generation slots: clarify/\w+/\w+ #0"):
        _pick_family_slot(plan, tracker, random.Random(0), closed={0, 1}, dead=dead)


def test_a_discarded_row_reserves_nothing_in_the_shared_tracker():
    from generators.pipeline import _StagedDuplicateTracker

    shared = DuplicateTracker(near_limit=1)
    spec = {"utterance": "turn on the lamp", "home": {}, "semantic_id": "lamp-on"}
    staged = _StagedDuplicateTracker(shared)
    staged.record(spec)
    assert staged.would_reject(spec) == "exact_duplicate_utterance"
    assert shared.would_reject(spec) is None

    fresh = _StagedDuplicateTracker(shared)
    assert fresh.would_reject(spec) is None
    fresh.record(spec)
    fresh.commit()
    assert shared.would_reject(spec) == "exact_duplicate_utterance"


def test_contrast_rows_carry_a_generated_grounding_family():
    rows = _sample()
    assert rows
    for row in rows:
        meta = row["metadata"]
        assert meta["grounding_family"] == f"ground_generated_{meta['contrast_axis']}"
    assert any(row["metadata"]["grounding_family"] == "ground_generated_base" for row in rows)


def test_contrast_rows_are_counted_in_the_grounding_stats():
    config = GeneratorConfig(
        count=120, seed=SEED, allocations={"contrast": 0.5, "ordinary": 0.5}, grounding_rate=0.1,
    )
    result = _run_generation(config)
    rows, report = result["rows"], result["stats"]["grounding"]
    contrast_rows = [row for row in rows if row["metadata"].get("family") == "contrast"]
    assert contrast_rows
    assert report["rows"] == sum(1 for row in rows if row["metadata"].get("grounding_family"))
    counted = {row["metadata"]["grounding_family"] for row in contrast_rows}
    assert counted <= set(report["by_family"])
    assert report["rows"] >= len(contrast_rows)


def test_tool_floors_the_donor_pool_cannot_reach_fail_at_plan_time():
    # Ordinary/settings slots are the only donors for tool floors; 40 of them cannot reach a 200 floor.
    with pytest.raises(ValueError, match="cannot be reserved"):
        build_plan(400, 3, {"ordinary": 0.1, "status": 0.9}, min_positive_per_tool=200)


def test_tool_floors_are_reserved_when_the_donor_pool_is_large_enough():
    from generators.capability_registry import CAPABILITIES, trainable_operations
    from generators.coverage import expected_tool

    plan = build_plan(
        4000, 3, {"ordinary": 0.5, "settings": 0.1, "status": 0.4},
        min_positive_per_tool=20, min_positive_per_operation=4,
    )
    carriers = {}
    for slot in plan.slots:
        if slot.family in {"ordinary", "settings"}:
            tool = expected_tool(slot.capability, slot.operation)
            carriers[tool] = carriers.get(tool, 0) + 1
    tools = {expected_tool(name, op.name) for name, cap in CAPABILITIES.items() for op in trainable_operations(cap)}
    for tool in tools - {None}:
        assert carriers.get(tool, 0) >= 20, (tool, carriers.get(tool, 0))
