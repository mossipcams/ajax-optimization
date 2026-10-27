from __future__ import annotations

from collections import Counter

import pytest

from generators.pairs import retarget_keeps_tool_floor

FLOOR = 200
SHORT_TOOL_SLOTS = [
    ("media_players", "unmute", "HassMediaPlayerUnmute"),
    ("media_players", "search_and_play", "HassMediaSearchAndPlay"),
    ("vacuums", "start", "HassVacuumStart"),
    ("vacuums", "return_home", "HassVacuumReturnToBase"),
]


def _slot(capability: str, operation: str) -> dict[str, str]:
    return {"family": "ordinary", "capability": capability, "operation": operation}


@pytest.mark.parametrize(("capability", "operation", "tool"), SHORT_TOOL_SLOTS)
def test_bias_blocked_while_slot_tool_below_floor(capability: str, operation: str, tool: str) -> None:
    slot = _slot(capability, operation)
    assert not retarget_keeps_tool_floor(slot, Counter(), FLOOR)
    assert not retarget_keeps_tool_floor(slot, Counter({tool: FLOOR - 1}), FLOOR)


@pytest.mark.parametrize(("capability", "operation", "tool"), SHORT_TOOL_SLOTS)
def test_bias_allowed_once_slot_tool_meets_floor(capability: str, operation: str, tool: str) -> None:
    slot = _slot(capability, operation)
    assert retarget_keeps_tool_floor(slot, Counter({tool: FLOOR}), FLOOR)


def test_other_tools_surplus_does_not_unlock_bias() -> None:
    slot = _slot("media_players", "unmute")
    assert not retarget_keeps_tool_floor(slot, Counter({"HassTurnOn": 5 * FLOOR}), FLOOR)


def test_bias_allowed_without_tool_floor() -> None:
    assert retarget_keeps_tool_floor(_slot("media_players", "unmute"), Counter(), 1)


def test_bias_allowed_for_slot_without_tool() -> None:
    assert retarget_keeps_tool_floor(_slot("no_such_capability", "no_such_operation"), Counter(), FLOOR)
