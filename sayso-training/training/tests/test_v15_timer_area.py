from __future__ import annotations

import json
import random
from pathlib import Path

from adapters.schema import tool_schema_map, validate_tool_arguments
from generators.tools import build_call_for_operation
from generators.utterances import _phrase_for_call

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
AREA_TIMER_OPERATIONS = ("pause", "unpause", "cancel", "status", "increase", "decrease")
AREAS = ("Living Room", "Kitchen", "Bedroom", "Office")
SEEDS = range(600)


def _schemas() -> dict:
    tools = json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))
    return tool_schema_map(tools)


def _calls(operation: str):
    for seed in SEEDS:
        area = AREAS[seed % len(AREAS)]
        yield area, build_call_for_operation(None, "timers", operation, random.Random(seed), area=area)


def test_area_timer_ops_emit_both_scoped_and_unscoped_calls() -> None:
    for operation in AREA_TIMER_OPERATIONS:
        scoped = [call for _, call in _calls(operation) if "area" in call["arguments"]]
        unscoped = [call for _, call in _calls(operation) if "area" not in call["arguments"]]
        assert scoped, operation
        assert unscoped, operation
        assert 0.28 <= len(scoped) / len(SEEDS) <= 0.42, operation


def test_area_scoped_timer_calls_carry_area_not_name() -> None:
    for operation in AREA_TIMER_OPERATIONS:
        for area, call in _calls(operation):
            arguments = call["arguments"]
            if "area" not in arguments:
                continue
            assert arguments["area"] == area
            assert arguments.get("name") not in AREAS


def test_area_scoped_timer_utterances_mention_area() -> None:
    for operation in AREA_TIMER_OPERATIONS:
        for seed, (area, call) in enumerate(_calls(operation)):
            if "area" not in call["arguments"]:
                continue
            utterance = _phrase_for_call("", call, str(seed))
            assert area.lower() in utterance.lower(), (operation, utterance)


def test_area_timer_calls_validate_against_pinned_schema() -> None:
    schemas = _schemas()
    for operation in AREA_TIMER_OPERATIONS:
        for _, call in _calls(operation):
            assert validate_tool_arguments(call["name"], call["arguments"], schemas) is None, call


def test_area_timer_calls_are_deterministic_for_a_seed() -> None:
    for operation in AREA_TIMER_OPERATIONS:
        first = build_call_for_operation(None, "timers", operation, random.Random(11), area="Kitchen")
        second = build_call_for_operation(None, "timers", operation, random.Random(11), area="Kitchen")
        assert first == second
