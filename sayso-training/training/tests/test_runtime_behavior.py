from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators.rendering import _round_time, render_example, scenario_to_spec
from generators.scenarios import build_scenario

TIMER_KEYS = [
    "id", "name", "device_id", "language", "start_hours", "start_minutes", "start_seconds", "is_active",
    "hours_left", "minutes_left", "seconds_left", "rounded_hours_left", "rounded_minutes_left",
    "rounded_seconds_left", "total_seconds_left",
]


def _render(utterance, *, calls=None, **scenario):
    spec = scenario_to_spec(build_scenario(**scenario))
    spec["utterance"] = utterance
    if calls is not None:
        spec["expected"] = {**spec["expected"], "kind": "action", "calls": calls}
    return render_example(spec)


def _tool_payload(row):
    tool_messages = [m for m in row["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    return json.loads(tool_messages[0]["content"])


def _spoken(row):
    return row["messages"][-1]["content"]


def test_datetime_row_reports_runtime_result_and_answers_with_time():
    row = _render(
        "what time is it", index=6, seed=19, capability="datetime", operation="query_time",
        home_size=8, family="datetime",
    )
    payload = _tool_payload(row)
    assert list(payload) == ["date", "time", "timezone", "weekday"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\d", payload["date"])
    assert re.fullmatch(r"\d\d:\d\d:\d\d", payload["time"])
    assert datetime.date.fromisoformat(payload["date"]).strftime("%A") == payload["weekday"]

    hour = int(payload["time"][:2])
    hour12 = hour % 12 or 12
    text = _spoken(row)
    assert text != "Done."
    assert f"{hour12}:{payload['time'][3:5]}" in text


def test_named_timer_status_row_reports_remaining_time():
    row = _render(
        "how much time is left on the pasta timer", index=1, seed=7, capability="timers", operation="status",
        home_size=8, calls=[{"name": "HassTimerStatus", "arguments": {"name": "pasta"}}],
    )
    payload = _tool_payload(row)
    assert list(payload) == ["speech", "response_type", "speech_slots", "data"]
    timers = payload["speech_slots"]["timers"]
    assert len(timers) == 1
    timer = timers[0]
    assert list(timer) == TIMER_KEYS
    assert timer["name"] == "pasta"
    assert timer["is_active"] is True

    text = _spoken(row)
    assert text != "Done."
    assert "pasta" in text
    assert str(timer["minutes_left"]) in text


def test_unnamed_timer_status_row_reports_remaining_time():
    row = _render(
        "how much time is left on my timer", index=2, seed=8, capability="timers", operation="status",
        home_size=8, calls=[{"name": "HassTimerStatus", "arguments": {}}],
    )
    timer = _tool_payload(row)["speech_slots"]["timers"][0]
    assert timer["name"] == ""
    assert _spoken(row) != "Done."


def test_round_time_matches_home_assistant_rounding():
    assert _round_time(0, 19, 52) == (0, 20, 0)
    assert _round_time(0, 19, 10) == (0, 19, 0)
    assert _round_time(0, 19, 30) == (0, 19, 30)
    assert _round_time(1, 50, 30) == (2, 0, 0)


def test_single_turn_on_action_still_ends_with_done_and_plain_result():
    row = _render(
        "turn on the den lights", index=1, seed=42, capability="lights", operation="turn_on", home_size=16,
    )
    call_message = next(m for m in row["messages"] if m.get("tool_calls"))
    assert call_message["tool_calls"][0]["function"]["name"] == "intent__HassTurnOn"
    assert _spoken(row) == "Done."
    payload = _tool_payload(row)
    assert set(payload) == {"speech", "response_type", "data"}
    assert payload["response_type"] == "action_done"
    assert payload["data"]["failed"] == []
