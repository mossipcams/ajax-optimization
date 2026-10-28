from collections import Counter
import json
import random
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.audit import audit_rows
from generators.config import GeneratorConfig, default_allocations
from generators.deduplication import DuplicateTracker
from generators.planning import (
    PRIMARY_FAMILIES,
    REAL_HOME_EXCLUDED_FAMILIES,
    CONDITIONAL_VARIANTS,
    CONVERSATION_CONTEXT_VARIANTS,
    RELATIVE_BRIGHTNESS_VARIANTS,
    RESTORE_OPERATIONS,
    TIMER_CONTEXT_VARIANTS,
    build_plan,
)
from generators.rendering import _tool_result, ha_brightness_pct, render_example, scenario_to_spec
from generators.homes import generate_home
from generators.row_generation import (
    _CONVERSATION_CONTEXT_CORRECT,
    _CONVERSATION_CONTEXT_HISTORY,
    _CONVERSATION_CONTEXT_OTHER,
    _CONVERSATION_CONTEXT_PRONOUN,
    _conversation_context_targets,
    _TIMER_CONTEXT_ADJUST_HISTORY,
    _TIMER_CONTEXT_BARE,
    _TIMER_CONTEXT_LOWER,
    _TIMER_CONTEXT_MINUTES,
    _TIMER_CONTEXT_NAMED,
    _TIMER_CONTEXT_NAMES,
    _TIMER_CONTEXT_RAISE,
    _TIMER_CONTEXT_START_NAMED,
    _RELATIVE_BRIGHTNESS_DELTAS,
    _RELATIVE_BRIGHTNESS_DOWN,
    _RELATIVE_BRIGHTNESS_READ,
    _RELATIVE_BRIGHTNESS_UP,
    _CONDITIONAL_READ,
    _CONDITIONAL_TEMPLATES,
    _RESTORE_BACK,
    _RESTORE_CHANGE,
    _RESTORE_READ,
    _TIMER_CONTEXT_START_UNNAMED,
    _conditional_plan,
    _relative_brightness_plan,
    _restore_plan,
    generate_row,
)
from generators.scenarios import build_scenario
from generators.tools import build_volume_relative, namespaced_tool_name
from generators.utterances import expand_utterance

TIMER_OPERATIONS = ("start", "pause", "unpause", "cancel", "status", "increase", "decrease")


def _timer_call(operation: str, index: int):
    scenario = build_scenario(index=index, seed=7, capability="timers", operation=operation, home_size=8)
    return scenario["expected"]["calls"][0]


def test_unnamed_timer_share_is_roughly_forty_percent():
    counts = {}
    for operation in TIMER_OPERATIONS:
        named = sum(1 for index in range(250) if _timer_call(operation, index)["arguments"].get("name"))
        counts[operation] = 1 - named / 250
    for operation, share in counts.items():
        assert 0.3 <= share <= 0.5, (operation, share)


def test_unnamed_timer_calls_carry_no_name_argument():
    seen_unnamed = {}
    for operation in TIMER_OPERATIONS:
        unnamed = [call for index in range(120) if not (call := _timer_call(operation, index))["arguments"].get("name")]
        assert unnamed, operation
        seen_unnamed[operation] = unnamed[0]
        for call in unnamed:
            assert "name" not in call["arguments"], operation


def test_unnamed_timer_utterances_never_mention_a_name():
    names = {"tea", "rice", "pasta", "bread", "coffee", "workout", "homework",
             "stretching", "watering", "roast", "cookies", "meditation"}
    for operation in TIMER_OPERATIONS:
        for index in range(60):
            scenario = build_scenario(index=index, seed=7, capability="timers", operation=operation, home_size=8)
            call = scenario["expected"]["calls"][0]
            if call["arguments"].get("name"):
                continue
            spec = {"expected": call and scenario["expected"], "home": scenario["home"],
                    "candidate_id": f"unnamed-{operation}-{index}", "target_names": [],
                    "capability": "timers", "operation": operation}
            utterance = expand_utterance(spec).casefold()
            assert not names & set(utterance.split()), (operation, index, utterance)


def test_unnamed_timer_templates_never_match_eval_utterances():
    from generators.utterances import _UNNAMED_TIMER_PHRASES, _duration
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for operation in TIMER_OPERATIONS:
        for index in range(40):
            call = _timer_call(operation, index)
            arguments = call["arguments"]
            if arguments.get("name") or arguments.get("area") or arguments.get("floor"):
                continue
            for template in _UNNAMED_TIMER_PHRASES[call["name"]]:
                rendered.add(template.format(delta=_duration(arguments)).casefold())
    overlap = rendered & eval_texts
    assert not overlap, overlap


def _volume_call(operation: str, index: int):
    scenario = build_scenario(index=index, seed=7, capability="media_players", operation=operation, home_size=8)
    return scenario["expected"]["calls"][0], scenario


def test_volume_step_is_minority_integer_in_range():
    steps_up = [_volume_call("volume_up", index)[0]["arguments"]["volume_step"] for index in range(250)]
    steps_down = [_volume_call("volume_down", index)[0]["arguments"]["volume_step"] for index in range(250)]
    all_steps = steps_up + steps_down
    integers = [step for step in all_steps if isinstance(step, int)]
    share = len(integers) / len(all_steps)
    assert 0.3 <= share <= 0.5, share
    for step in integers:
        assert 5 <= abs(step) <= 30, step
    for step in steps_up:
        if isinstance(step, int):
            assert step > 0, step
    for step in steps_down:
        if isinstance(step, int):
            assert step < 0, step
    assert {"up", "down"} & set(all_steps), all_steps[:10]


def test_volume_step_varies_per_row_for_one_entity():
    entity = {"name": "Living Room TV", "domain": "media_player"}
    steps = [
        build_volume_relative(entity, direction="up", rng=random.Random(index))["arguments"]["volume_step"]
        for index in range(150)
    ]
    integers = [step for step in steps if isinstance(step, int)]
    words = [step for step in steps if not isinstance(step, int)]
    assert integers and words, (len(integers), len(words))
    assert all(word == "up" for word in words), words
    assert len({abs(step) for step in integers}) >= 3, integers
    again = [
        build_volume_relative(entity, direction="up", rng=random.Random(index))["arguments"]["volume_step"]
        for index in range(150)
    ]
    assert steps == again


def test_volume_step_draw_does_not_consume_shared_rng():
    entity = {"name": "Living Room TV", "domain": "media_player"}
    rng = random.Random(1234)
    repeated = [
        build_volume_relative(entity, direction="up", rng=rng)["arguments"]["volume_step"]
        for _ in range(5)
    ]
    assert len(set(repeated)) == 1, repeated


def test_integer_volume_utterance_number_matches_step():
    for operation in ("volume_up", "volume_down"):
        seen_integers = 0
        for index in range(80):
            call, scenario = _volume_call(operation, index)
            step = call["arguments"]["volume_step"]
            if not isinstance(step, int):
                continue
            seen_integers += 1
            spec = {"expected": scenario["expected"], "home": scenario["home"],
                    "candidate_id": f"volstep-{operation}-{index}",
                    "target_names": [call["arguments"]["name"]],
                    "capability": "media_players", "operation": operation}
            utterance = expand_utterance(spec).casefold()
            assert str(abs(step)) in utterance, (operation, index, step, utterance)
            if step > 0:
                assert any(word in utterance for word in ("up", "loud", "increase", "raise")), utterance
            else:
                assert any(word in utterance for word in ("down", "quiet", "decrease", "lower")), utterance
        assert seen_integers >= 10, (operation, seen_integers)


def test_volume_step_templates_never_match_eval_utterances():
    from generators.utterances import _VOLUME_STEP_PHRASES, _VOLUME_UP_PHRASES, _VOLUME_DOWN_PHRASES
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for target in ("the tv", "tv", "the living room tv", "the kitchen satellite"):
        for template in (*_VOLUME_UP_PHRASES, *_VOLUME_DOWN_PHRASES):
            rendered.add(template.format(target).casefold())
        for direction in ("up", "down"):
            for n in (5, 10, 15, 20, 25, 30):
                for template in _VOLUME_STEP_PHRASES[direction]:
                    rendered.add(template.format(target=target, n=n).casefold())
    overlap = rendered & eval_texts
    assert not overlap, overlap


def _tool_result_messages(row):
    return [m for m in row["messages"] if m.get("role") == "tool"]


def _named_timer_start_spec():
    for index in range(200):
        spec = scenario_to_spec(build_scenario(index=index, seed=20260517, capability="timers", operation="start", home_size=16))
        if spec["expected"]["calls"][0]["arguments"].get("name"):
            return spec
    raise AssertionError("no named timer start in 200 scenarios")


def test_intent_tool_result_uses_as_dict_shape():
    spec = _named_timer_start_spec()
    name = spec["expected"]["calls"][0]["arguments"]["name"]
    spec["utterance"] = f"Start the {name} timer."
    row = render_example(spec)
    tool_messages = _tool_result_messages(row)
    assert len(tool_messages) == 1
    result = json.loads(tool_messages[0]["content"])
    assert set(result) == {"speech", "response_type", "data"}
    assert result["response_type"] == "action_done"
    assert result["speech"] == {}
    # A timer name is not a home entity, so nothing is listed as a success target.
    assert result["data"] == {"success": [], "failed": []}


def test_intent_success_lists_the_targeted_entity_with_its_entity_id():
    scenario = build_scenario(index=11, seed=20260531, capability="lights", operation="query_state", home_size=16)
    spec = scenario_to_spec(scenario)
    light = next(e for e in spec["home"]["entities"] if e["capability"] == "lights")
    spec["spoken_targets"] = {light["name"]: "the porch light"}
    call = {"name": "HassLightSet", "arguments": {"name": light["name"], "domain": ["light"], "brightness": 40}}
    result = _tool_result(spec, call)
    assert result == {"speech": {}, "response_type": "action_done", "data": {
        "success": [{"name": "the porch light", "type": "entity", "id": light["entity_id"]}], "failed": []}}


def test_get_live_context_tool_result_includes_attributes():
    scenario = build_scenario(index=11, seed=20260531, capability="lights", operation="query_state", home_size=16)
    spec = scenario_to_spec(scenario)
    name = spec["target_names"][0]
    light = next(e for e in spec["home"]["entities"] if e["name"] == name)
    light["attributes"] = {"brightness": "51"}
    player = {**light, "name": "Kitchen Satellite", "domain": "media_player", "attributes": {"volume_level": "0.3"}}
    spec["home"]["entities"].append(player)
    spec["target_names"] = [name, player["name"]]
    spec["utterance"] = f"What is the state of {name}?"
    row = render_example(spec)
    tool_messages = _tool_result_messages(row)
    assert len(tool_messages) == 1
    result = json.loads(tool_messages[0]["content"])
    assert set(result) == {"result"}
    text = result["result"]
    assert text.startswith("Live Context: An overview of the areas and the devices in this smart home:\n")
    assert f"- names: {name}" in text
    assert "attributes:\n    brightness: '51'\n    brightness_pct: '20'" in text
    assert f"names: {player['name']}" in text
    assert "volume_level: 0.3" in text


def test_ha_brightness_pct_matches_home_assistant_rounding():
    # HA: str(max(round(b/255*100), 1) if b > 0 else round(b/255*100))
    assert ha_brightness_pct(0) == 0
    assert ha_brightness_pct(1) == 1
    assert ha_brightness_pct(3) == 1
    assert ha_brightness_pct(128) == 50
    assert ha_brightness_pct(255) == 100


def test_get_live_context_light_matches_home_assistant_example():
    from generators.rendering import _live_attributes

    shaped = _live_attributes({"brightness": 3})
    assert shaped == {"brightness": "3", "brightness_pct": "1"}
    assert _live_attributes({"volume_level": "0.1"}) == {"volume_level": 0.1}


def _satellite_home(area="Living Room"):
    from generators.homes import make_entity

    sat = make_entity(name=f"{area} Satellite", capability="media_players", area=area,
                      floor="Main Floor", rng=random.Random(1), aliases=[f"{area} Satellite"])
    tv = make_entity(name=f"{area} TV", capability="media_players", area=area,
                     floor="Main Floor", rng=random.Random(2), aliases=[f"{area} TV"])
    return {"home_id": "home_satellite_test", "sayso_entity_area": area, "entities": [sat, tv],
            "areas": [area]}, sat, tv


def _volume_spec(home, calls, excluded=(), seed="sat"):
    expected = {"kind": "action", "calls": list(calls),
                "data": {"targets": [], "success": [], "failed": []}}
    return {"candidate_id": f"sat_{seed}", "category": "exclusion" if excluded else "normal",
            "user_utterance": "", "home": home, "spoken_targets": {},
            "expected": expected, "excluded_names": list(excluded),
            "target_names": [call["arguments"]["name"] for call in calls]}


def test_satellite_share_in_generated_homes():
    from generators.homes import generate_home
    from generators.capability_registry import is_satellite

    rng = random.Random(20261130)
    with_satellite = 0
    for index in range(80):
        home = generate_home(index, 24, rng)
        satellites = [e for e in home["entities"] if is_satellite(e)]
        assert len(satellites) <= 1
        if satellites:
            with_satellite += 1
            sat = satellites[0]
            assert sat["area"] == home["sayso_entity_area"]
            assert sat["name"] in {f"{home['sayso_entity_area']} Satellite", "SaySo Satellite"}, sat["name"]
            assert sat["device_class"] == "speaker", sat
    share = with_satellite / 80
    assert 0.2 <= share <= 0.4, share


def test_generated_homes_are_deterministic_for_seed():
    from generators.homes import generate_home

    first = generate_home(3, 24, random.Random(9))
    second = generate_home(3, 24, random.Random(9))
    assert first["entities"] == second["entities"]
    assert first["context_entities"] == second["context_entities"]


def test_satellite_has_no_absolute_volume():
    from generators.capability_registry import entity_supports
    from generators.homes import media_player_profile

    _, features = media_player_profile("SaySo Satellite")
    assert "volume" not in features and "volume_step" in features
    home, sat, _ = _satellite_home()
    assert entity_supports(sat, "media_players", "volume_up")
    assert not entity_supports(sat, "media_players", "volume_set")


def test_generated_scenarios_never_set_absolute_satellite_volume():
    checked = 0
    for index in range(200):
        scenario = build_scenario(index=index, seed=20261020, capability="media_players",
                                  operation="volume_set", home_size=24)
        calls = scenario["expected"]["calls"]
        satellites = {e["name"] for e in scenario["home"]["entities"] if e["name"].endswith(" Satellite")
                      or e["name"] == "SaySo Satellite"}
        assert not [c for c in calls if c["arguments"].get("name") in satellites], calls
        checked += 1
    assert checked == 200


def test_self_reference_volume_rows_target_satellite_relative_only():
    home, sat, _ = _satellite_home()
    rel_up = {"name": "HassSetVolumeRelative", "arguments": {"name": sat["name"], "volume_step": "up"}}
    rel_down = {"name": "HassSetVolumeRelative", "arguments": {"name": sat["name"], "volume_step": "down"}}
    for index, call in enumerate((rel_up, rel_down)):
        text = expand_utterance(_volume_spec(home, [call], seed=f"self{index} ")).casefold()
        assert sat["name"].casefold() not in text
        assert "percent" not in text and "your volume" not in text, text
    up_text = expand_utterance(_volume_spec(home, [rel_up], seed="up")).casefold()
    assert any(phrase in up_text for phrase in ("speak up", "turn yourself up", "make yourself a little louder")), up_text
    down_text = expand_utterance(_volume_spec(home, [rel_down], seed="down")).casefold()
    assert any(phrase in down_text for phrase in ("you're a bit too loud", "turn yourself down", "quiet down")), down_text


def test_contrast_rows_carry_the_other_device_in_excluded_names():
    from generators.utterances import _SELF_VOLUME_DOWN

    home, sat, tv = _satellite_home()
    self_down = {"name": "HassSetVolumeRelative",
                 "arguments": {"name": sat["name"], "volume_step": "down"}}
    spec = _volume_spec(home, [self_down], excluded=(tv["name"],), seed="contrast-sat")
    assert spec["excluded_names"] == [tv["name"]]
    text = expand_utterance(spec).casefold()
    assert text.endswith(f", not the {tv['name']}".casefold()), text
    assert any(text.startswith(phrase) for phrase in _SELF_VOLUME_DOWN), text
    assert "leave" not in text and sat["name"] not in text, text
    tv_down = {"name": "HassSetVolumeRelative",
               "arguments": {"name": tv["name"], "volume_step": "down"}}
    spec_reverse = _volume_spec(home, [tv_down], excluded=(sat["name"],), seed="contrast-tv")
    assert spec_reverse["excluded_names"] == [sat["name"]]
    text_reverse = expand_utterance(spec_reverse).casefold()
    assert text_reverse.endswith(", not you"), text_reverse
    assert tv["name"].casefold() in text_reverse, text_reverse
    assert "leave" not in text_reverse and sat["name"].casefold() not in text_reverse, text_reverse


def test_contrast_rows_use_contrast_phrasing_not_leave_alone():
    home, sat, tv = _satellite_home()
    calls = [
        {"name": "HassSetVolumeRelative", "arguments": {"name": sat["name"], "volume_step": "up"}},
        {"name": "HassSetVolumeRelative", "arguments": {"name": sat["name"], "volume_step": "down"}},
    ]
    for index, call in enumerate(calls):
        text = expand_utterance(_volume_spec(home, [call], excluded=(tv["name"],), seed=f"no-leave{index}"))
        assert ", but leave" not in text.casefold(), text
        assert text.casefold().endswith(f", not the {tv['name']}".casefold()), text


def test_contrast_clause_needs_exactly_one_satellite_and_one_other_device():
    home, sat, tv = _satellite_home()
    self_down = {"name": "HassSetVolumeRelative", "arguments": {"name": sat["name"], "volume_step": "down"}}
    two_excluded = _volume_spec(home, [self_down], excluded=(tv["name"], "Hallway Lamp"), seed="two")
    assert ", but leave " in expand_utterance(two_excluded)
    same_device = _volume_spec(home, [self_down], excluded=(sat["name"],), seed="same")
    assert ", but leave " in expand_utterance(same_device)


def test_satellite_rows_from_generated_scenarios_target_the_satellite():
    from generators.utterances import _SELF_VOLUME_DOWN, _SELF_VOLUME_UP

    seen = 0
    for index in range(400):
        scenario = build_scenario(index=index, seed=20261020, capability="media_players",
                                  operation="volume_down", home_size=24)
        call = scenario["expected"]["calls"][0]
        satellites = [e["name"] for e in scenario["home"]["entities"] if e.get("name", "").endswith(" Satellite")]
        if call["arguments"]["name"] not in satellites:
            continue
        seen += 1
        text = expand_utterance(scenario_to_spec(scenario)).casefold()
        assert call["arguments"]["name"].casefold() not in text, text
        assert any(phrase in text for phrase in (*_SELF_VOLUME_DOWN, *_SELF_VOLUME_UP)) or "percent" in text, text
    assert seen >= 5, seen


def test_self_volume_templates_never_match_eval_utterances():
    from generators.utterances import _SELF_VOLUME_DOWN, _SELF_VOLUME_UP

    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for phrase in (*_SELF_VOLUME_UP, *_SELF_VOLUME_DOWN):
        rendered.add(phrase.casefold())
    for step in (-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30):
        rendered.add(f"turn yourself {'up' if step > 0 else 'down'} {abs(step)} percent")
    overlap = rendered & eval_texts
    assert not overlap, overlap


def _satellite_tv_exclusion_scenario(index: int):
    home, sat, tv = _satellite_home()
    scenario = build_scenario(index=index, seed=20261020, capability="media_players",
                              operation="volume_down", home_size=24, robustness="exclusion", home=home)
    return scenario, sat, tv


def test_satellite_tv_exclusion_builds_contrast_rows_in_both_directions():
    counts = {"satellite": 0, "tv": 0}
    for index in range(80):
        scenario, sat, tv = _satellite_tv_exclusion_scenario(index)
        if not scenario["excluded_names"]:
            continue
        calls = scenario["expected"]["calls"]
        assert len(calls) == 1, calls
        target = calls[0]["arguments"]["name"]
        assert target in {sat["name"], tv["name"]}, target
        other = tv["name"] if target == sat["name"] else sat["name"]
        assert scenario["excluded_names"] == [other], scenario["excluded_names"]
        counts["satellite" if target == sat["name"] else "tv"] += 1
    assert counts["satellite"] >= 10 and counts["tv"] >= 10, counts


def test_satellite_tv_contrast_rows_pass_validate_row():
    from generators.validation import validate_row

    checked = 0
    for index in range(80):
        scenario, sat, tv = _satellite_tv_exclusion_scenario(index)
        if not scenario["excluded_names"]:
            continue
        spec = scenario_to_spec(scenario)
        spec["utterance"] = expand_utterance(spec)
        reason = validate_row(spec)
        assert reason is None, (spec["utterance"], reason)
        excluded = scenario["excluded_names"][0]
        if excluded == sat["name"]:
            assert spec["utterance"].endswith(", not you"), spec["utterance"]
        else:
            assert spec["utterance"].endswith(f", not the {tv['name']}"), spec["utterance"]
        checked += 1
    assert checked >= 10, checked


def test_exclusion_row_missing_excluded_device_fails_validation():
    from generators.validation import validate_spec, validate_utterance

    home, sat, tv = _satellite_home()
    self_down = {"name": "HassSetVolumeRelative", "arguments": {"name": sat["name"], "volume_step": "down"}}
    spec = _volume_spec(home, [self_down], excluded=(tv["name"],), seed="omit")
    assert validate_spec(spec) is None
    spec["utterance"] = "turn yourself down a notch"
    assert validate_utterance(spec) == "missing_exclusion"
    spec["utterance"] = "turn yourself down a notch, not the Hallway Lamp"
    assert validate_utterance(spec) == "missing_exclusion"
    spec["utterance"] = "turn yourself down a notch, not the TV"
    assert validate_utterance(spec) == "missing_exclusion"
    spec["utterance"] = "turn yourself down a notch, not the " + tv["name"]
    assert validate_utterance(spec) is None


def test_single_call_exclusion_without_satellite_contrast_still_needs_two_calls():
    from generators.validation import validate_spec

    home, sat, tv = _satellite_home()
    tv_down = {"name": "HassSetVolumeRelative", "arguments": {"name": tv["name"], "volume_step": "down"}}
    spec = _volume_spec(home, [tv_down], excluded=("Hallway Lamp",), seed="plain")
    assert validate_spec(spec) == "missing_multiple_actions"


def _timer_context_slot(variant: str, index: int) -> dict:
    return {"index": index, "family": "timer_context", "robustness": "ambiguity", "capability": "timers",
            "operation": variant, "tier": 1, "home_size": 16, "area_scenario": None, "grounding": False}


def _timer_context_row(variant: str, index: int = 0, seed: int = 20261020) -> dict:
    config = GeneratorConfig(count=10, seed=seed)
    tracker = DuplicateTracker()
    reason = None
    for attempt in range(20):
        row, reason = generate_row(_timer_context_slot(variant, index), config,
                                   random.Random(seed + index * 31 + attempt),
                                   excluded=set(), dup_tracker=tracker, attempt=attempt)
        if row is not None:
            return row
    raise AssertionError((variant, index, reason))


def _timer_context_rows(variant: str, count: int = 40) -> list[dict]:
    return [_timer_context_row(variant, index) for index in range(count)]


def _history_turns(row: dict) -> int:
    turns = (len(row["messages"]) - 5) // 3
    assert len(row["messages"]) == 5 + 3 * turns, len(row["messages"])
    return turns


def _history_calls(row: dict) -> list[dict]:
    return [
        {
            "name": row["messages"][2 + 3 * turn]["tool_calls"][0]["function"]["name"],
            "arguments": json.loads(row["messages"][2 + 3 * turn]["tool_calls"][0]["function"]["arguments"]),
        }
        for turn in range(_history_turns(row))
    ]


def _final_call(row: dict) -> dict:
    function = row["messages"][-3]["tool_calls"][0]["function"]
    return {"name": function["name"], "arguments": json.loads(function["arguments"])}


def _tool_payload(row: dict, position: int) -> dict:
    return json.loads(row["messages"][position]["content"])


def test_timer_context_is_selectable_without_changing_default_allocations():
    assert "timer_context" not in PRIMARY_FAMILIES
    assert set(default_allocations()) == set(PRIMARY_FAMILIES)
    default_plan = build_plan(120, 7, default_allocations())
    assert set(default_plan.requested) == set(PRIMARY_FAMILIES)
    plan = build_plan(60, 7, {"timer_context": 0.5, "ordinary": 0.5})
    assert plan.requested == {"timer_context": 30, "ordinary": 30}
    timer_slots = [slot for slot in plan.slots if slot.family == "timer_context"]
    assert len(timer_slots) == 30
    for slot in timer_slots:
        assert (slot.capability, slot.robustness) == ("timers", "ambiguity")
        assert slot.operation in TIMER_CONTEXT_VARIANTS


def test_timer_context_slots_cover_every_variant():
    plan = build_plan(400, 11, {"timer_context": 1.0})
    assert {slot.operation for slot in plan.slots} == set(TIMER_CONTEXT_VARIANTS)


def test_timer_context_message_order_and_train_flags():
    for variant in TIMER_CONTEXT_VARIANTS:
        for row in _timer_context_rows(variant, count=12):
            messages = row["messages"]
            turns = _history_turns(row)
            assert 1 <= turns <= 2, (variant, turns)
            assert messages[0]["role"] == "system" and messages[0]["train_on_turn"] is False
            for turn in range(turns):
                user, call, tool = messages[1 + 3 * turn: 4 + 3 * turn]
                assert user["role"] == "user" and user["train_on_turn"] is False
                assert call["role"] == "assistant" and call["train_on_turn"] is False
                assert len(call["tool_calls"]) == 1
                assert call["tool_calls"][0]["function"]["name"] in {
                    namespaced_tool_name("HassStartTimer"),
                    namespaced_tool_name("HassIncreaseTimer"),
                }
                assert tool["role"] == "tool" and tool["train_on_turn"] is False
                assert tool["tool_call_id"] == call["tool_calls"][0]["id"]
            final_user, final_call, final_tool, reply = messages[-4:]
            assert final_user["role"] == "user" and final_user["train_on_turn"] is False
            assert final_call["role"] == "assistant" and final_call["train_on_turn"] is True
            assert len(final_call["tool_calls"]) == 1
            assert final_tool["role"] == "tool" and final_tool["train_on_turn"] is False
            assert final_tool["tool_call_id"] == final_call["tool_calls"][0]["id"]
            assert reply["role"] == "assistant" and reply["train_on_turn"] is True
            assert "tool_calls" not in reply
            assert row["metadata"]["family"] == "timer_context"


def test_bare_command_variant_calls_unnamed_timer_tool():
    rows = _timer_context_rows("bare_command")
    assert {_history_turns(row) for row in rows} == {1}
    for row in rows:
        final = _final_call(row)
        assert "name" not in final["arguments"]
        assert final["name"] in {
            namespaced_tool_name(tool)
            for tool in ("HassPauseTimer", "HassUnpauseTimer", "HassCancelTimer", "HassIncreaseTimer", "HassDecreaseTimer")
        }
        assert _tool_payload(row, -2)["response_type"] == "action_done"
        assert row["messages"][-1]["content"] == "Done."
        assert row["metadata"]["expected_target_names"] == [
            call["arguments"]["name"] for call in _history_calls(row) if "name" in call["arguments"]
        ]


def test_ambiguous_timers_variant_returns_multiple_error_and_names_both_timers():
    rows = _timer_context_rows("ambiguous_timers")
    for row in rows:
        assert _history_turns(row) == 2
        first_name, second_name = (call["arguments"]["name"] for call in _history_calls(row))
        assert first_name != second_name
        assert "name" not in _final_call(row)["arguments"]
        assert _tool_payload(row, -2) == {"error": "MultipleTimersMatchedError", "error_text": "Multiple timers matched"}
        assert row["messages"][-1]["content"] == f"Did you mean the {first_name} timer or the {second_name} timer?"
        assert row["messages"][-1]["train_on_turn"] is True


def test_correction_variant_raises_or_lowers_by_the_difference():
    seen = set()
    for row in _timer_context_rows("correction", count=60):
        final = _final_call(row)
        history = _history_calls(row)
        assert final["arguments"] == {"minutes": 5}
        assert row["messages"][-1]["content"] == "Done."
        if final["name"] == namespaced_tool_name("HassIncreaseTimer"):
            seen.add("raise")
            assert len(history) == 1
            assert history[0]["name"] == namespaced_tool_name("HassStartTimer")
            assert history[0]["arguments"]["minutes"] == 10
        else:
            seen.add("lower")
            assert final["name"] == namespaced_tool_name("HassDecreaseTimer")
            assert len(history) == 2
            assert history[0]["arguments"]["minutes"] == 10
            assert history[1] == {"name": namespaced_tool_name("HassIncreaseTimer"), "arguments": {"minutes": 10}}
    assert seen == {"raise", "lower"}


def test_correction_utterances_keep_their_leading_no_without_a_politeness_wrapper():
    for variant in TIMER_CONTEXT_VARIANTS:
        for row in _timer_context_rows(variant, count=60):
            assert "could you no," not in row["messages"][-4]["content"].casefold()
    for row in _timer_context_rows("correction", count=60):
        assert row["messages"][-4]["content"].casefold().startswith("no,")


def test_missing_timer_variant_returns_not_found_and_refuses_by_name():
    for row in _timer_context_rows("missing_timer"):
        history_name = _history_calls(row)[0]["arguments"]["name"]
        missing = _final_call(row)["arguments"]["name"]
        assert missing != history_name
        assert missing in row["messages"][-4]["content"]
        assert _tool_payload(row, -2) == {"error": "TimerNotFoundError", "error_text": "Timer not found"}
        assert row["messages"][-1]["content"] == f"There's no {missing} timer running."
        assert row["messages"][-1]["train_on_turn"] is True
        assert row["metadata"]["expected_target_names"] == [history_name, missing]


def test_timer_context_rows_are_deterministic_and_pass_audit():
    for variant in TIMER_CONTEXT_VARIANTS:
        assert json.dumps(_timer_context_row(variant, index=5), sort_keys=True) == json.dumps(
            _timer_context_row(variant, index=5), sort_keys=True
        )
        rows = [_timer_context_row(variant, index) for index in range(25)]
        assert audit_rows(rows)["rows"] == len(rows)


def test_timer_context_templates_never_match_eval_utterances():
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for templates in (*_TIMER_CONTEXT_BARE.values(), _TIMER_CONTEXT_RAISE, _TIMER_CONTEXT_LOWER):
        for template in templates:
            for minutes in (2, 3, 4, 5):
                rendered.add(template.format(minutes=minutes).casefold())
    for templates in _TIMER_CONTEXT_NAMED.values():
        for template in templates:
            for name in _TIMER_CONTEXT_NAMES:
                for minutes in (2, 3, 4, 5):
                    rendered.add(template.format(name=name, minutes=minutes).casefold())
    for template in (*_TIMER_CONTEXT_START_NAMED, *_TIMER_CONTEXT_START_UNNAMED, *_TIMER_CONTEXT_ADJUST_HISTORY):
        for minutes in _TIMER_CONTEXT_MINUTES:
            for name in _TIMER_CONTEXT_NAMES:
                rendered.add(template.format(name=name, minutes=minutes).casefold())
    overlap = rendered & eval_texts
    assert not overlap, overlap


def test_media_remote_and_app_operations_are_refused():
    from generators.capability_registry import SupportLevel, operation_spec

    for operation in ("navigate", "launch_app"):
        spec = operation_spec("media_players", operation)
        assert spec is not None
        assert spec.support == SupportLevel.UNAVAILABLE
        assert spec.blocker


def test_tv_show_phrasing_maps_to_pause_and_unpause():
    from generators.utterances import _TV_SHOW_PHRASES, _phrase_for_call

    tv_pause = {"name": "HassMediaPause", "arguments": {"name": "Den TV", "domain": ["media_player"], "device_class": ["tv"]}}
    tv_play = {"name": "HassMediaUnpause", "arguments": {"name": "Den TV", "domain": ["media_player"], "device_class": ["tv"]}}
    speaker_pause = {"name": "HassMediaPause", "arguments": {"name": "Den Satellite", "domain": ["media_player"], "device_class": ["speaker"]}}
    pause_phrases = set()
    play_phrases = set()
    for index in range(200):
        pause_phrases.add(_phrase_for_call("the den TV", tv_pause, f"seed:{index}"))
        play_phrases.add(_phrase_for_call("the den TV", tv_play, f"seed:{index}"))
        assert "TV" not in _phrase_for_call("the den satellite", speaker_pause, f"seed:{index}")
    expected_pause = {phrase.format(target="the den TV") for phrase in _TV_SHOW_PHRASES["HassMediaPause"]}
    expected_play = {phrase.format(target="the den TV") for phrase in _TV_SHOW_PHRASES["HassMediaUnpause"]}
    assert expected_pause <= pause_phrases
    assert expected_play <= play_phrases
    assert all("the den TV" in phrase for phrase in pause_phrases | play_phrases)


def test_tv_show_phrasing_always_names_the_target_tv():
    from generators.utterances import _TV_SHOW_PHRASES

    for phrases in _TV_SHOW_PHRASES.values():
        for phrase in phrases:
            assert "{target}" in phrase, phrase


def test_tv_show_templates_never_match_eval_utterances():
    from generators.utterances import _TV_SHOW_PHRASES

    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = {
        phrase.format(target="the den TV").casefold()
        for phrases in _TV_SHOW_PHRASES.values() for phrase in phrases
    }
    overlap = rendered & eval_texts
    assert not overlap, overlap


def test_blocked_media_operations_generate_refusals_not_exceptions():
    for operation in ("navigate", "launch_app"):
        requests = set()
        for index in range(30):
            with patch("generators.row_generation.pick_robustness", return_value="unavailable"):
                row, reason = generate_row(
                    {"index": index, "capability": "media_players", "operation": operation, "home_size": 8},
                    GeneratorConfig(), random.Random(index), excluded=set(), dup_tracker=DuplicateTracker(),
                )
            assert row is not None, (operation, reason)
            assert not any(message.get("tool_calls") for message in row["messages"]), operation
            assert "available Home Assistant tools" in row["messages"][-1]["content"]
            requests.add(next(m["content"] for m in row["messages"] if m["role"] == "user").casefold())
        assert requests, operation


def test_unavailable_request_templates_cover_new_media_and_cover_operations():
    from generators.scenarios.unavailable import _UNAVAILABLE_REQUESTS

    for key in (("media_players", "navigate"), ("media_players", "launch_app"), ("covers", "set_position")):
        templates = _UNAVAILABLE_REQUESTS[key]
        assert isinstance(templates, tuple) and len(templates) >= 2, key
    assert "halfway" in " ".join(_UNAVAILABLE_REQUESTS[("covers", "set_position")])
    assert any("30 percent" in t for t in _UNAVAILABLE_REQUESTS[("covers", "set_position")])


def test_unavailable_request_templates_never_match_eval_utterances():
    from generators.scenarios.unavailable import _UNAVAILABLE_REQUESTS

    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for value in _UNAVAILABLE_REQUESTS.values():
        for template in (value if isinstance(value, tuple) else (value,)):
            rendered.add(template.format(name="the den TV").casefold())
    overlap = rendered & eval_texts
    assert not overlap, overlap

_BRIGHTNESS_TEST_NAMES = ("Kitchen Light", "Desk Lamp", "Bedroom Lamp", "Hallway Ceiling Lights")


def _relative_brightness_slot(variant: str, index: int) -> dict:
    return {"index": index, "family": "relative_brightness", "robustness": "ordinary", "capability": "lights",
            "operation": variant, "tier": 1, "home_size": 16, "area_scenario": None, "grounding": False}


def _relative_brightness_row(variant: str, index: int = 0, seed: int = 20261020) -> dict:
    config = GeneratorConfig(count=10, seed=seed)
    tracker = DuplicateTracker()
    reason = None
    for attempt in range(20):
        row, reason = generate_row(_relative_brightness_slot(variant, index), config,
                                   random.Random(seed + index * 31 + attempt),
                                   excluded=set(), dup_tracker=tracker, attempt=attempt)
        if row is not None:
            return row
    raise AssertionError((variant, index, reason))


def _relative_brightness_rows(variant: str, count: int = 30) -> list[dict]:
    return [_relative_brightness_row(variant, index) for index in range(count)]


def _brightness_target(raw: int, delta: int) -> int:
    """HA's brightness_pct is what the model reads back; the label clamps that percent plus the delta."""
    return min(100, max(0, int(str(ha_brightness_pct(raw))) + delta))


def _template_delta(utterance: str, name: str) -> int:
    """Signed delta whose unvaried template renders exactly to `utterance`."""
    for delta in _RELATIVE_BRIGHTNESS_DELTAS:
        templates = _RELATIVE_BRIGHTNESS_UP if delta > 0 else _RELATIVE_BRIGHTNESS_DOWN
        if any(template.format(name=name, amount=abs(delta)) == utterance for template in templates):
            return delta
    raise AssertionError(utterance)


def _spoken_delta(utterance: str) -> int:
    """Signed delta read from a rendered row utterance (casing and prefixes may vary)."""
    amount = int(re.search(r"(\d+) percent", utterance).group(1))
    return -amount if re.search(r"\b(dim|down|lower)\b", utterance.casefold()) else amount


def test_relative_brightness_is_selectable_and_excluded_from_real_homes():
    assert "relative_brightness" not in PRIMARY_FAMILIES
    assert "relative_brightness" in REAL_HOME_EXCLUDED_FAMILIES
    assert set(default_allocations()) == set(PRIMARY_FAMILIES)
    plan = build_plan(60, 7, {"relative_brightness": 0.5, "ordinary": 0.5})
    assert plan.requested == {"relative_brightness": 30, "ordinary": 30}
    slots = [slot for slot in plan.slots if slot.family == "relative_brightness"]
    assert len(slots) == 30
    for slot in slots:
        assert (slot.capability, slot.robustness) == ("lights", "ordinary")
        assert slot.operation in RELATIVE_BRIGHTNESS_VARIANTS


def test_relative_brightness_slots_cover_every_variant():
    plan = build_plan(400, 11, {"relative_brightness": 1.0})
    assert {slot.operation for slot in plan.slots} == set(RELATIVE_BRIGHTNESS_VARIANTS)


def test_relative_brightness_deltas_are_explicit_integer_percents():
    assert set(_RELATIVE_BRIGHTNESS_DELTAS) == {-50, -40, -30, -25, -20, -10, 10, 20, 25, 30, 40, 50}
    for template in (*_RELATIVE_BRIGHTNESS_UP, *_RELATIVE_BRIGHTNESS_DOWN):
        assert "{amount} percent" in template
        assert "little" not in template and "bit" not in template


def test_relative_brightness_label_is_clamped_brightness_plus_delta():
    seen = set()
    for seed in range(400):
        plan = _relative_brightness_plan("set_from_context", "Desk Lamp", random.Random(seed))
        raw = plan["raw_brightness"]
        assert 0 <= raw <= 255
        delta = _template_delta(plan["utterance"], "Desk Lamp")
        target = plan["call"]["arguments"]["brightness"]
        assert target == _brightness_target(raw, delta)
        assert 0 <= target <= 100
        unclamped = ha_brightness_pct(raw) + delta
        if unclamped < 0:
            seen.add("floor")
            assert target == 0
        if unclamped > 100:
            seen.add("ceiling")
            assert target == 100
    assert seen == {"floor", "ceiling"}


def test_relative_brightness_clamp_arithmetic_at_both_ends():
    assert _brightness_target(0, -10) == 0
    assert _brightness_target(0, -50) == 0
    assert _brightness_target(255, 10) == 100
    assert _brightness_target(255, 50) == 100
    assert _brightness_target(128, 20) == 70
    assert _brightness_target(128, -30) == 20


def test_read_first_variant_reads_state_then_scores_like_status():
    for row in _relative_brightness_rows("read_first"):
        messages = row["messages"]
        assert len(messages) == 5, len(messages)
        assert messages[0]["role"] == "system" and messages[0]["train_on_turn"] is False
        user, call, tool, reply = messages[1:]
        assert user["role"] == "user" and user["train_on_turn"] is False
        assert call["role"] == "assistant" and call["train_on_turn"] is True
        assert len(call["tool_calls"]) == 1
        assert tool["role"] == "tool" and tool["train_on_turn"] is False
        assert tool["tool_call_id"] == call["tool_calls"][0]["id"]
        assert reply["role"] == "assistant" and reply["train_on_turn"] is True and "tool_calls" not in reply
        final = _final_call(row)
        name = final["arguments"]["name"]
        assert final == {"name": namespaced_tool_name("GetLiveContext"), "arguments": {"name": name, "domain": "light"}}
        assert name.casefold() in user["content"].casefold()
        assert reply["content"] == f"{name} is on."
        assert row["metadata"]["family"] == "relative_brightness"
        assert row["metadata"]["expected_target_names"] == [name]


def test_set_from_context_variant_reads_then_sets_clamped_brightness():
    for row in _relative_brightness_rows("set_from_context"):
        messages = row["messages"]
        assert len(messages) == 8, len(messages)
        system, history_user, history_call, history_tool, user, final_call, final_tool, reply = messages
        assert system["train_on_turn"] is False
        for message in (history_user, history_call, history_tool):
            assert message["train_on_turn"] is False
        read_call = history_call["tool_calls"][0]["function"]
        assert read_call["name"] == namespaced_tool_name("GetLiveContext")
        read_args = json.loads(read_call["arguments"])
        name = read_args["name"]
        assert read_args == {"name": name, "domain": "light"}
        history_result = _tool_payload(row, 3)
        assert set(history_result) == {"result"}
        raw = int(re.search(r"brightness: '(\d+)'", history_result["result"]).group(1))
        assert user["train_on_turn"] is False and name.casefold() in user["content"].casefold()
        assert final_call["train_on_turn"] is True
        assert final_call["tool_calls"][0]["function"]["name"] == namespaced_tool_name("HassLightSet")
        final = _final_call(row)
        assert set(final["arguments"]) == {"name", "domain", "brightness"}
        assert final["arguments"]["name"] == name
        assert final["arguments"]["brightness"] == _brightness_target(raw, _spoken_delta(user["content"]))
        assert final_tool["train_on_turn"] is False and final_tool["tool_call_id"] == final_call["tool_calls"][0]["id"]
        assert reply["train_on_turn"] is True and "tool_calls" not in reply
        assert reply["content"] == f"{name} is now at {final['arguments']['brightness']} percent."
        assert row["metadata"]["expected_target_names"] == [name]


def test_relative_brightness_templates_never_match_eval_utterances():
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for templates in (_RELATIVE_BRIGHTNESS_UP, _RELATIVE_BRIGHTNESS_DOWN):
        for template in templates:
            for name in _BRIGHTNESS_TEST_NAMES:
                for amount in _RELATIVE_BRIGHTNESS_DELTAS:
                    rendered.add(template.format(name=name, amount=abs(amount)).casefold())
    for template in _RELATIVE_BRIGHTNESS_READ:
        for name in _BRIGHTNESS_TEST_NAMES:
            rendered.add(template.format(name=name).casefold())
    overlap = rendered & eval_texts
    assert not overlap, overlap


def test_relative_brightness_rows_are_deterministic_and_pass_audit():
    for variant in RELATIVE_BRIGHTNESS_VARIANTS:
        assert json.dumps(_relative_brightness_row(variant, index=5), sort_keys=True) == json.dumps(
            _relative_brightness_row(variant, index=5), sort_keys=True
        )
        rows = [_relative_brightness_row(variant, index) for index in range(20)]
        assert audit_rows(rows)["rows"] == len(rows)


def _conversation_context_slot(variant: str, index: int) -> dict:
    return {"index": index, "family": "conversation_context", "robustness": "ambiguity", "capability": "lights",
            "operation": variant, "tier": 1, "home_size": 16, "area_scenario": None, "grounding": False}


def _conversation_context_row(variant: str, index: int = 0, seed: int = 20261020) -> dict:
    config = GeneratorConfig(count=10, seed=seed)
    tracker = DuplicateTracker()
    reason = None
    for attempt in range(20):
        row, reason = generate_row(_conversation_context_slot(variant, index), config,
                                   random.Random(seed + index * 31 + attempt),
                                   excluded=set(), dup_tracker=tracker, attempt=attempt)
        if row is not None:
            return row
    raise AssertionError((variant, index, reason))


def _conversation_context_rows(variant: str, count: int = 40) -> list[dict]:
    return [_conversation_context_row(variant, index) for index in range(count)]


def _operation_of(tool_name: str) -> str:
    return {namespaced_tool_name("HassTurnOn"): "turn_on", namespaced_tool_name("HassTurnOff"): "turn_off"}[tool_name]


def test_conversation_context_is_selectable_and_excluded_from_real_homes():
    assert "conversation_context" not in PRIMARY_FAMILIES
    assert "conversation_context" in REAL_HOME_EXCLUDED_FAMILIES
    assert set(default_allocations()) == set(PRIMARY_FAMILIES)
    plan = build_plan(60, 7, {"conversation_context": 0.5, "ordinary": 0.5})
    assert plan.requested == {"conversation_context": 30, "ordinary": 30}
    slots = [slot for slot in plan.slots if slot.family == "conversation_context"]
    assert len(slots) == 30
    for slot in slots:
        assert slot.robustness == "ambiguity"
        assert slot.capability in {"lights", "switches"}
        assert slot.operation in CONVERSATION_CONTEXT_VARIANTS


def test_conversation_context_slots_cover_every_variant():
    plan = build_plan(400, 11, {"conversation_context": 1.0})
    assert {slot.operation for slot in plan.slots} == set(CONVERSATION_CONTEXT_VARIANTS)


def test_conversation_context_message_order_and_train_flags():
    for variant in CONVERSATION_CONTEXT_VARIANTS:
        for row in _conversation_context_rows(variant, count=20):
            messages = row["messages"]
            assert _history_turns(row) == 1
            assert messages[0]["role"] == "system" and messages[0]["train_on_turn"] is False
            user, call, tool = messages[1:4]
            assert user["role"] == "user" and user["train_on_turn"] is False
            assert call["role"] == "assistant" and call["train_on_turn"] is False
            assert tool["role"] == "tool" and tool["train_on_turn"] is False
            assert tool["tool_call_id"] == call["tool_calls"][0]["id"]
            final_user, final_call, final_tool, reply = messages[-4:]
            assert final_user["role"] == "user" and final_user["train_on_turn"] is False
            assert final_call["role"] == "assistant" and final_call["train_on_turn"] is True
            assert len(final_call["tool_calls"]) == 1
            assert final_tool["role"] == "tool" and final_tool["train_on_turn"] is False
            assert final_tool["tool_call_id"] == final_call["tool_calls"][0]["id"]
            assert reply["role"] == "assistant" and reply["train_on_turn"] is True
            assert "tool_calls" not in reply
            assert reply["content"].endswith(" is now on.") or reply["content"].endswith(" is now off.")
            assert row["metadata"]["family"] == "conversation_context"
            assert row["metadata"]["real_home"] is False


def test_previous_turn_reference_acts_on_the_history_device_with_a_pronoun():
    rows = _conversation_context_rows("previous_turn_reference")
    assert len(rows) == 40
    for row in rows:
        history_name = _history_calls(row)[0]["arguments"]["name"]
        history_op = _operation_of(_history_calls(row)[0]["name"])
        final = _final_call(row)
        assert final["arguments"] == {"name": history_name}
        final_op = _operation_of(final["name"])
        assert final_op != history_op
        assert history_name.casefold() in row["messages"][1]["content"].casefold()
        assert history_name.casefold() not in row["messages"][-4]["content"].casefold()
        assert row["messages"][-1]["content"] == f"{history_name} is now {'on' if final_op == 'turn_on' else 'off'}."
        assert row["metadata"]["expected_target_names"] == [history_name]


def test_other_one_acts_on_the_second_device_in_the_same_area():
    rows = _conversation_context_rows("other_one")
    for row in rows:
        history_name = _history_calls(row)[0]["arguments"]["name"]
        final = _final_call(row)
        assert final["arguments"].keys() == {"name"}
        assert final["arguments"]["name"] != history_name
        assert _operation_of(final["name"]) == _operation_of(_history_calls(row)[0]["name"])
        assert "other one" in row["messages"][-4]["content"].casefold()
        assert row["metadata"]["expected_target_names"] == [final["arguments"]["name"]]


def test_other_one_targets_are_a_same_area_pair():
    for seed in range(20):
        home = generate_home(seed, 24, random.Random(seed))
        entities = [entity for entity in home["entities"] if entity["capability"] == "lights"]
        targets = _conversation_context_targets("other_one", entities, random.Random(seed))
        area_counts = Counter(entity["area"] for entity in entities)
        if targets is None:
            assert not any(count == 2 for count in area_counts.values())
            continue
        history_entity, target = targets
        assert history_entity is not target
        assert history_entity["area"] == target["area"]
        assert area_counts[history_entity["area"]] == 2


def test_other_one_never_fires_in_an_area_with_three_same_capability_devices():
    crowded = [
        {"name": name, "area": "Kitchen", "capability": "lights"}
        for name in ("Ceiling", "Counter", "Island")
    ]
    for seed in range(20):
        assert _conversation_context_targets("other_one", crowded, random.Random(seed)) is None
    mixed = crowded + [
        {"name": "Study Lamp", "area": "Study", "capability": "lights"},
        {"name": "Desk Lamp", "area": "Study", "capability": "lights"},
    ]
    for seed in range(20):
        history_entity, target = _conversation_context_targets("other_one", mixed, random.Random(seed))
        assert history_entity["area"] == target["area"] == "Study"
        assert {history_entity["name"], target["name"]} == {"Study Lamp", "Desk Lamp"}


def test_target_correction_repeats_the_operation_on_the_new_device_only():
    for row in _conversation_context_rows("target_correction"):
        history = _history_calls(row)[0]
        final = _final_call(row)
        assert final["name"] == history["name"]
        assert final["arguments"]["name"] != history["arguments"]["name"]
        assert len(row["messages"]) == 8
        assert row["messages"][-4]["content"].casefold().endswith(final["arguments"]["name"].casefold())
        assert row["messages"][-1]["content"] == f"{final['arguments']['name']} is now {'on' if _operation_of(final['name']) == 'turn_on' else 'off'}."
        assert row["metadata"]["expected_target_names"] == [final["arguments"]["name"]]


def test_conversation_context_rows_are_deterministic_and_pass_audit():
    for variant in CONVERSATION_CONTEXT_VARIANTS:
        assert json.dumps(_conversation_context_row(variant, index=5), sort_keys=True) == json.dumps(
            _conversation_context_row(variant, index=5), sort_keys=True
        )
        rows = [_conversation_context_row(variant, index) for index in range(20)]
        assert audit_rows(rows)["rows"] == len(rows)


def test_conversation_context_templates_never_match_eval_utterances():
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = set()
    for templates in (*_CONVERSATION_CONTEXT_HISTORY.values(), _CONVERSATION_CONTEXT_CORRECT):
        for template in templates:
            for name in _BRIGHTNESS_TEST_NAMES:
                rendered.add(template.format(name=name).casefold())
    for templates in (*_CONVERSATION_CONTEXT_PRONOUN.values(), _CONVERSATION_CONTEXT_OTHER):
        for template in templates:
            rendered.add(template.casefold())
    overlap = rendered & eval_texts
    assert not overlap, overlap



def _multi_action_scenarios(count: int):
    scenarios = [
        build_scenario(index=index, seed=29, capability="media_players", operation="mute",
                       home_size=32, targeting="multiple", robustness="multi_action")
        for index in range(count)
    ]
    return [scenario for scenario in scenarios if "target_entities" in scenario]


def _timer_pair_utterance(scenario) -> str:
    spec = scenario_to_spec(scenario)
    return expand_utterance({**spec, "category": "multi_action"})


def test_timer_pairs_join_multi_action_at_roughly_a_fifth():
    multi = _multi_action_scenarios(300)
    pairs = [scenario for scenario in multi if scenario.get("timer_companion")]
    assert pairs
    assert 0.12 <= len(pairs) / len(multi) <= 0.28, (len(pairs), len(multi))


def test_timer_pair_label_has_one_device_call_then_one_timer_in_spoken_order():
    for scenario in _multi_action_scenarios(300):
        if not scenario.get("timer_companion"):
            continue
        calls = scenario["expected"]["calls"]
        names = [call["name"] for call in calls]
        assert len(calls) == 2, names
        assert names.count("HassStartTimer") == 1, names
        assert names[0] != "HassStartTimer", names
        device_part, joiner, timer_part = _timer_pair_utterance(scenario).casefold().partition(" and ")
        assert joiner and "timer" not in device_part and "timer" in timer_part, device_part + joiner + timer_part


def test_timer_pair_utterance_mentions_the_timer_duration():
    for scenario in _multi_action_scenarios(300):
        if not scenario.get("timer_companion"):
            continue
        timer = scenario["expected"]["calls"][-1]
        amounts = [str(timer["arguments"][unit]) for unit in ("hours", "minutes", "seconds") if timer["arguments"].get(unit)]
        assert amounts, timer
        assert any(amount in _timer_pair_utterance(scenario) for amount in amounts), _timer_pair_utterance(scenario)


def test_timer_pairs_are_named_or_unnamed_at_the_timer_rate():
    pairs = [scenario for scenario in _multi_action_scenarios(900) if scenario.get("timer_companion")]
    assert len(pairs) >= 100, len(pairs)
    unnamed = sum(1 for scenario in pairs if not scenario["expected"]["calls"][-1]["arguments"].get("name"))
    assert 0.25 <= unnamed / len(pairs) <= 0.55, (unnamed, len(pairs))


def test_timer_pair_utterances_never_match_eval_utterances():
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    eval_texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                eval_texts.add(text)
    rendered = {
        _timer_pair_utterance(scenario).casefold().rstrip("?.!")
        for scenario in _multi_action_scenarios(300)
        if scenario.get("timer_companion")
    }
    assert rendered
    overlap = rendered & eval_texts
    assert not overlap, overlap



_RESTORE_TEST_NAMES = ("Kitchen Light", "Desk Lamp", "Living Room TV", "Craft Room Satellite")
_CONDITIONAL_TEST_NAMES = ("Kitchen Light", "Desk Lamp", "Guest Fan", "Porch Outlet")
_RESTORE_CAPABILITY_BY_OPERATION = {operation: capability for capability, operation in RESTORE_OPERATIONS}


def _restore_slot(operation: str, index: int) -> dict:
    return {"index": index, "family": "restore", "robustness": "ordinary",
            "capability": _RESTORE_CAPABILITY_BY_OPERATION[operation], "operation": operation,
            "tier": 1, "home_size": 16, "area_scenario": None, "grounding": False}


def _conditional_slot(variant: str, index: int) -> dict:
    return {"index": index, "family": "conditional", "robustness": "ordinary", "capability": "lights",
            "operation": variant, "tier": 1, "home_size": 16, "area_scenario": None, "grounding": False}


def _family_rows(make_slot, variant: str, count: int, seed: int = 20261020) -> list[dict]:
    """Rows from one batch: shared duplicate and casing trackers, as the pipeline keeps them."""
    config = GeneratorConfig(count=10, seed=seed)
    tracker = DuplicateTracker()
    casing_counts: dict[str, Counter[str]] = {"call": Counter(), "no_call": Counter()}
    rows = []
    for index in range(count):
        reason = None
        for attempt in range(20):
            row, reason = generate_row(make_slot(variant, index), config,
                                       random.Random(seed + index * 31 + attempt),
                                       excluded=set(), dup_tracker=tracker, attempt=attempt,
                                       casing_counts=casing_counts)
            if row is not None:
                break
        if row is None:
            raise AssertionError((variant, index, reason))
        has_calls = any(message.get("tool_calls") and message["train_on_turn"] for message in row["messages"])
        first_user = row["messages"][1]["content"]
        casing_counts["call" if has_calls else "no_call"]["upper" if first_user[:1].isupper() else "lower"] += 1
        rows.append(row)
    return rows


def _restore_rows(operation: str, count: int = 30) -> list[dict]:
    return _family_rows(_restore_slot, operation, count)


def _conditional_rows(variant: str, count: int = 30) -> list[dict]:
    return _family_rows(_conditional_slot, variant, count)


def _eval_utterances() -> set[str]:
    eval_root = Path(__file__).resolve().parents[2] / "vendor" / "SaySo" / "evals" / "cases"
    texts = set()
    for path in sorted(eval_root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("utterance") or "").casefold().rstrip("?.!")
            if text:
                texts.add(text)
    return texts


def _read_state(row: dict) -> str:
    # The history or trained GetLiveContext result is at index 3 in every multi-turn row here.
    return _tool_payload(row, 3)["result"].split("state: '")[1].split("'")[0]


def test_restore_and_conditional_are_selectable_and_excluded_from_real_homes():
    for family in ("restore", "conditional"):
        assert family not in PRIMARY_FAMILIES
        assert family in REAL_HOME_EXCLUDED_FAMILIES
    assert set(default_allocations()) == set(PRIMARY_FAMILIES)
    plan = build_plan(60, 7, {"restore": 0.5, "conditional": 0.5})
    assert plan.requested == {"restore": 30, "conditional": 30}
    for slot in plan.slots:
        if slot.family == "restore":
            assert (slot.capability, slot.operation) in RESTORE_OPERATIONS
            assert slot.robustness == "ordinary"
        else:
            assert slot.capability in {"lights", "switches", "fans"}
            assert slot.operation in CONDITIONAL_VARIANTS


def test_restore_and_conditional_slots_cover_every_variant():
    plan = build_plan(400, 11, {"restore": 0.5, "conditional": 0.5})
    assert {(slot.capability, slot.operation) for slot in plan.slots if slot.family == "restore"} == set(RESTORE_OPERATIONS)
    assert {slot.operation for slot in plan.slots if slot.family == "conditional"} == set(CONDITIONAL_VARIANTS)


def test_restore_brightness_label_is_the_percent_that_was_read():
    for seed in range(300):
        plan = _restore_plan("lights", "set_brightness", "Desk Lamp", random.Random(seed))
        raw = int(plan["attributes"]["brightness"])
        assert 0 <= raw <= 255
        assert plan["call"] == {"name": "HassLightSet", "arguments": {
            "name": "Desk Lamp", "domain": ["light"], "brightness": ha_brightness_pct(raw)}}
        assert 0 <= plan["call"]["arguments"]["brightness"] <= 100
        assert plan["history"][1]["call"]["arguments"]["brightness"] != plan["call"]["arguments"]["brightness"]


def test_restore_volume_label_is_the_level_that_was_read():
    for seed in range(300):
        plan = _restore_plan("media_players", "volume_set", "Living Room TV", random.Random(seed))
        level = float(plan["attributes"]["volume_level"])
        assert 0.0 <= level <= 1.0
        assert plan["call"] == {"name": "HassSetVolume", "arguments": {
            "name": "Living Room TV", "domain": ["media_player"], "volume_level": round(level * 100)}}
        assert plan["history"][1]["call"]["arguments"]["volume_level"] != plan["call"]["arguments"]["volume_level"]


def test_restore_label_arithmetic_at_both_ends_of_the_brightness_scale():
    assert round(0 / 255 * 100) == 0
    assert round(128 / 255 * 100) == 50
    assert round(255 / 255 * 100) == 100


def test_restore_history_reads_then_changes_then_restores():
    for _, operation in RESTORE_OPERATIONS:
        change_tool = "HassLightSet" if operation == "set_brightness" else "HassSetVolume"
        for row in _restore_rows(operation, count=20):
            messages = row["messages"]
            assert len(messages) == 11, len(messages)
            assert messages[0]["role"] == "system" and messages[0]["train_on_turn"] is False
            (read_user, read_call, read_tool, change_user, change_call, change_result,
             back_user, final_call, final_tool, reply) = messages[1:]
            for message in (read_user, read_tool, change_user, change_result, back_user, final_tool):
                assert message["train_on_turn"] is False
            assert read_call["train_on_turn"] is False and change_call["train_on_turn"] is False
            assert final_call["train_on_turn"] is True
            assert reply["train_on_turn"] is True and "tool_calls" not in reply
            assert read_call["tool_calls"][0]["function"]["name"] == namespaced_tool_name("GetLiveContext")
            assert change_call["tool_calls"][0]["function"]["name"] == namespaced_tool_name(change_tool)
            assert change_result["tool_call_id"] == change_call["tool_calls"][0]["id"]
            assert final_tool["tool_call_id"] == final_call["tool_calls"][0]["id"]
            final = _final_call(row)
            assert final["name"] == namespaced_tool_name(change_tool)
            assert final["arguments"]["name"] == json.loads(change_call["tool_calls"][0]["function"]["arguments"])["name"]
            assert row["metadata"]["family"] == "restore"
            assert row["metadata"]["expected_target_names"] == [final["arguments"]["name"]]


def test_restore_trained_call_restores_the_value_read_from_history():
    for row in _restore_rows("set_brightness", count=30):
        raw = int(re.search(r"brightness: '(\d+)'", _tool_payload(row, 3)["result"]).group(1))
        final = _final_call(row)
        assert final["name"] == namespaced_tool_name("HassLightSet")
        assert final["arguments"]["brightness"] == ha_brightness_pct(raw)
        assert row["messages"][-1]["content"] == (
            f"{final['arguments']['name']} is now at {final['arguments']['brightness']} percent.")
    for row in _restore_rows("volume_set", count=30):
        level = float(re.search(r"volume_level: ([\d.]+)", _tool_payload(row, 3)["result"]).group(1))
        final = _final_call(row)
        assert final["name"] == namespaced_tool_name("HassSetVolume")
        assert final["arguments"]["volume_level"] == round(level * 100)


def test_restore_trained_utterance_is_a_restore_phrase():
    for _, operation in RESTORE_OPERATIONS:
        for row in _restore_rows(operation, count=30):
            name = _final_call(row)["arguments"]["name"]
            lowered = row["messages"][7]["content"].casefold()
            assert any(template.format(name=name).casefold().rstrip(".") in lowered for template in _RESTORE_BACK)


def test_conditional_read_first_trains_a_status_read_of_the_condition_target():
    for row in _conditional_rows("read_first"):
        messages = row["messages"]
        assert len(messages) == 5, len(messages)
        system, user, call, tool, reply = messages
        assert system["train_on_turn"] is False and user["train_on_turn"] is False
        assert call["train_on_turn"] is True and tool["train_on_turn"] is False
        assert reply["train_on_turn"] is True and "tool_calls" not in reply
        final = _final_call(row)
        name = final["arguments"]["name"]
        assert final["name"] == namespaced_tool_name("GetLiveContext")
        assert set(final["arguments"]) == {"name", "domain"}
        assert name.casefold() in user["content"].casefold()
        assert reply["content"] == f"{name} is {_read_state(row)}."
        assert row["metadata"]["family"] == "conditional"
        assert row["metadata"]["expected_target_names"] == [name]


def test_conditional_condition_holds_runs_the_action_after_reading_the_state():
    for row in _conditional_rows("condition_holds"):
        messages = row["messages"]
        assert len(messages) == 8, len(messages)
        system, read_user, read_call, read_tool, user, call, tool, reply = messages
        for message in (read_user, read_tool, user, tool):
            assert message["train_on_turn"] is False
        assert read_call["train_on_turn"] is False and call["train_on_turn"] is True
        assert reply["train_on_turn"] is True and "tool_calls" not in reply
        assert read_call["tool_calls"][0]["function"]["name"] == namespaced_tool_name("GetLiveContext")
        state = _read_state(row)
        final = _final_call(row)
        name = final["arguments"]["name"]
        expected_tool = "HassTurnOff" if state == "on" else "HassTurnOn"
        assert final["name"] == namespaced_tool_name(expected_tool)
        assert reply["content"] == f"{name} is now {'off' if expected_tool == 'HassTurnOff' else 'on'}."
        assert row["metadata"]["expected_target_names"] == [name]


def test_conditional_condition_fails_speaks_the_state_without_a_call():
    for row in _conditional_rows("condition_fails"):
        messages = row["messages"]
        assert len(messages) == 6, len(messages)
        system, read_user, read_call, read_tool, user, reply = messages
        assert read_call["train_on_turn"] is False
        assert user["role"] == "user" and user["train_on_turn"] is False
        assert reply["role"] == "assistant" and reply["train_on_turn"] is True and "tool_calls" not in reply
        assert not any("tool_calls" in message for message in messages[4:])
        name = row["metadata"]["expected_target_names"][0]
        assert reply["content"] == f"{name} is already {_read_state(row)}."
        assert row["metadata"]["no_action_reason"] == "already_in_state"


def test_conditional_branches_cover_both_actions_and_both_states():
    seen = set()
    for row in _conditional_rows("condition_holds", count=40) + _conditional_rows("condition_fails", count=40):
        seen.add((row["metadata"]["operation"], _read_state(row)))
    assert seen == {("turn_off", "on"), ("turn_on", "off"), ("turn_off", "off"), ("turn_on", "on")}


def test_conditional_templates_never_match_eval_utterances():
    rendered = set()
    for templates in _CONDITIONAL_TEMPLATES.values():
        for template in templates:
            for name in _CONDITIONAL_TEST_NAMES:
                rendered.add(template.format(name=name).casefold().rstrip("?.!"))
    for template in _CONDITIONAL_READ:
        for name in _CONDITIONAL_TEST_NAMES:
            rendered.add(template.format(name=name).casefold().rstrip("?.!"))
    overlap = rendered & _eval_utterances()
    assert not overlap, overlap


def test_restore_templates_never_match_eval_utterances():
    rendered = set()
    for templates in _RESTORE_READ.values():
        for template in templates:
            for name in _RESTORE_TEST_NAMES:
                rendered.add(template.format(name=name).casefold().rstrip("?.!"))
    for templates in _RESTORE_CHANGE.values():
        for template in templates:
            for name in _RESTORE_TEST_NAMES:
                for amount in (10, 50, 100):
                    rendered.add(template.format(name=name, amount=amount).casefold().rstrip("?.!"))
    for template in _RESTORE_BACK:
        for name in _RESTORE_TEST_NAMES:
            rendered.add(template.format(name=name).casefold().rstrip("?.!"))
    overlap = rendered & _eval_utterances()
    assert not overlap, overlap


def test_conditional_plan_label_follows_condition_and_variant():
    entity = {"name": "Desk Lamp", "domain": "light", "capability": "lights"}
    for seed in range(200):
        holds = _conditional_plan("condition_holds", entity, random.Random(seed))
        fails = _conditional_plan("condition_fails", entity, random.Random(seed))
        assert holds["state"] == holds["condition"]
        assert fails["state"] != fails["condition"]
        assert holds["action"] == ("turn_off" if holds["condition"] == "on" else "turn_on")
        assert fails["action"] == holds["action"]
        assert holds["action_call"]["name"] == ("HassTurnOff" if holds["action"] == "turn_off" else "HassTurnOn")
        assert len(holds["history"]) == 1 and holds["history"][0]["call"]["name"] == "GetLiveContext"
        assert _conditional_plan("read_first", entity, random.Random(seed))["history"] == []


def test_restore_and_conditional_rows_are_deterministic_and_pass_audit():
    for _, operation in RESTORE_OPERATIONS:
        assert json.dumps(_restore_rows(operation, count=6), sort_keys=True) == json.dumps(
            _restore_rows(operation, count=6), sort_keys=True
        )
        rows = _restore_rows(operation, count=20)
        assert audit_rows(rows)["rows"] == len(rows)
    for variant in CONDITIONAL_VARIANTS:
        rows = _conditional_rows(variant, count=20)
        assert audit_rows(rows)["rows"] == len(rows)


def test_v8_recipe_allocates_every_new_family_and_planning_accepts_them():
    recipe = Path(__file__).resolve().parents[2] / "training" / "configs" / "generation" / "full_sft_v8.yaml"
    cfg = GeneratorConfig.from_yaml(recipe, repo_root=Path(__file__).resolve().parents[2])
    new_families = {"timer_context", "conversation_context", "relative_brightness", "restore", "conditional"}
    assert new_families <= set(cfg.allocations)
    assert abs(sum(cfg.allocations.values()) - 1.0) < 1e-9
    assert cfg.allocations["contrast"] == 0.105
    plan = build_plan(2000, cfg.seed, cfg.allocations)
    for family in new_families | {"contrast"}:
        assert plan.requested[family] > 0, family
    assert {slot.family for slot in plan.slots} >= new_families | {"contrast"}


def test_blocked_refusal_operations_scale_with_the_capability_quota():
    from generators.capability_registry import CAPABILITIES
    from generators.sampling import compute_operation_quotas

    cap = CAPABILITIES["media_players"]
    small = compute_operation_quotas(100, cap)
    large = compute_operation_quotas(12000, cap)
    for operation in ("navigate", "launch_app"):
        assert small[operation] == 1
        assert large[operation] >= 20


def test_blocked_request_operations_reach_the_unsupported_family_for_v8():
    from dataclasses import asdict

    from generators.capability_registry import SupportLevel, operation_spec
    from generators.planning import BLOCKED_REQUEST_OPERATIONS
    from generators.scenarios.unavailable import _UNAVAILABLE_REQUESTS

    for key in BLOCKED_REQUEST_OPERATIONS:
        assert key in _UNAVAILABLE_REQUESTS, key
        assert operation_spec(*key).support == SupportLevel.UNAVAILABLE, key

    root = Path(__file__).resolve().parents[2]
    cfg = GeneratorConfig.from_yaml(root / "training" / "configs" / "generation" / "full_sft_v8.yaml", repo_root=root)
    plan = build_plan(2000, cfg.seed, cfg.allocations)
    unsupported = [slot for slot in plan.slots if slot.family == "unsupported"]
    blocked = [slot for slot in unsupported if (slot.capability, slot.operation) in BLOCKED_REQUEST_OPERATIONS]
    assert {(slot.capability, slot.operation) for slot in blocked} == set(BLOCKED_REQUEST_OPERATIONS)
    assert 0.05 <= len(blocked) / len(unsupported) <= 0.25

    sampled = {}
    for slot in blocked:
        sampled.setdefault((slot.capability, slot.operation), slot)
    for key, slot in sampled.items():
        spec = asdict(slot)
        row, reason = generate_row(
            {**spec, "robustness": "unavailable"}, cfg, random.Random(slot.index),
            excluded=set(), dup_tracker=DuplicateTracker(),
        )
        assert row is not None, (key, reason)
        assert not any(message.get("tool_calls") for message in row["messages"]), key
        assert "available Home Assistant tools" in row["messages"][-1]["content"], key
