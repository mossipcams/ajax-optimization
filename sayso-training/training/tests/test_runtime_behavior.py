from __future__ import annotations

import copy
import datetime
import json
import random
import re
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators import shared_alias
from generators.capability_registry import CAPABILITIES, entity_supports, trainable_operations
from generators.context import system_prompt
from generators.tools import production_catalog
from generators.real_home import load_real_home
from generators.rendering import _round_time, render_example, scenario_to_spec
from generators.scenarios import build_scenario
from generators.tools import build_light_set
from generators.validation import gate_named_device_calls

TIMER_KEYS = [
    "id", "name", "device_id", "language", "start_hours", "start_minutes", "start_seconds", "is_active",
    "hours_left", "minutes_left", "seconds_left", "rounded_hours_left", "rounded_minutes_left",
    "rounded_seconds_left", "total_seconds_left",
]

REAL_HOME_PATH = TRAINING_ROOT / "fixtures" / "real_home.json"
TARGETINGS = ("individual", "area", "floor", "multiple")
BANNED_LIGHT_COLORS = ("warm white", "cool white", "soft white", "daylight")


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


def _tool_calls(scenario):
    return scenario["expected"].get("calls") or []


def test_real_home_floor_arguments_are_only_fixture_floors():
    fixture = json.loads(REAL_HOME_PATH.read_text(encoding="utf-8"))
    fixture_floors = set(fixture["area_floors"].values()) | {entity["floor"] for entity in fixture["entities"]}
    assert fixture_floors == {"Floor 1"}

    floors = set()
    for capability, spec in CAPABILITIES.items():
        for operation in trainable_operations(spec):
            for targeting in TARGETINGS:
                for index in range(12):
                    scenario = build_scenario(
                        index=index, seed=3, capability=capability, operation=operation.name,
                        home_size=8, targeting=targeting, home=load_real_home(REAL_HOME_PATH),
                    )
                    floors |= {
                        call["arguments"]["floor"] for call in _tool_calls(scenario)
                        if "floor" in call["arguments"]
                    }
    assert floors
    assert floors <= fixture_floors


def test_volume_step_only_media_player_gets_no_relative_volume_label():
    home = load_real_home(REAL_HOME_PATH)
    media_players = [entity for entity in home["entities"] if entity["capability"] == "media_players"]
    assert media_players
    for entity in media_players:
        entity["capabilities"] = ["on", "off", "volume_step", "mute"]
        assert not entity_supports(entity, "media_players", "volume_up")
        assert not entity_supports(entity, "media_players", "volume_down")

    for operation in ("volume_up", "volume_down"):
        for targeting in ("individual", "area"):
            for index in range(12):
                scenario = build_scenario(
                    index=index, seed=11, capability="media_players", operation=operation,
                    home_size=8, targeting=targeting, inject_missing=False,
                    home=copy.deepcopy(home),
                )
                names = {call["name"] for call in _tool_calls(scenario)}
                assert "HassSetVolumeRelative" not in names


def test_generated_light_color_arguments_are_home_assistant_names():
    colors = {
        build_light_set({"name": "Lamp"}, random.Random(seed), "set_color")["arguments"]["color"]
        for seed in range(200)
    }
    assert colors
    assert not colors & set(BANNED_LIGHT_COLORS)

    for index in range(12):
        scenario = build_scenario(index=index, seed=5, capability="lights", operation="set_color", home_size=16)
        for call in _tool_calls(scenario):
            if call["name"] == "HassLightSet":
                assert call["arguments"].get("color") not in BANNED_LIGHT_COLORS

    for template in shared_alias._TEMPLATES["set_color"]:
        assert not any(color in template for color in BANNED_LIGHT_COLORS)


LABEL_HOME = {
    "areas": ["Den", "Living Room"],
    "entities": [
        {"name": "Den Ceiling Lights", "domain": "light", "capability": "lights", "capabilities": ["on", "off", "brightness", "color"],
         "area": "Den", "aliases": [], "entity_id": "light.den_ceiling_lights"},
        {"name": "Living Room TV", "domain": "media_player", "capability": "media_players",
         "capabilities": ["on", "off"], "area": "Living Room", "aliases": [],
         "entity_id": "media_player.living_room_tv"},
        {"name": "Sun", "domain": "sun", "capability": None, "capabilities": [], "area": None,
         "aliases": [], "entity_id": "sun.sun"},
        {"name": "Den Temperature", "domain": "sensor", "capability": None, "capabilities": [],
         "area": "Den", "aliases": [], "entity_id": "sensor.den_temperature"},
    ],
}


def _label_row(call_name, arguments, utterance="turn on the den lights"):
    return {
        "messages": [
            {"role": "system", "content": system_prompt(LABEL_HOME, utterance), "train_on_turn": False},
            {"role": "user", "content": utterance, "train_on_turn": False},
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": True,
                "tool_calls": [{
                    "id": "call_0", "type": "function",
                    "function": {"name": call_name, "arguments": json.dumps(arguments)},
                }],
            },
        ],
        "tools": production_catalog(LABEL_HOME),
        "metadata": {},
    }


def test_gate_target_consistency_rejects_name_outside_area_and_wrong_domain():
    outside = _label_row("intent__HassTurnOn", {"name": "Den Ceiling Lights", "area": "Living Room"})
    assert gate_named_device_calls(outside) == "target_outside_area"

    wrong_domain = _label_row("intent__HassTurnOn", {"name": "Den Ceiling Lights", "domain": ["switch"]})
    assert gate_named_device_calls(wrong_domain) == "target_domain_mismatch"


def test_gate_target_consistency_accepts_name_in_area_and_domain():
    row = _label_row("intent__HassTurnOn", {"name": "Den Ceiling Lights", "area": "Den", "domain": ["light"]})
    assert gate_named_device_calls(row) is None


def test_gate_unactionable_domain_rejects_turn_on_of_sun():
    row = _label_row("intent__HassTurnOn", {"name": "Sun"})
    assert gate_named_device_calls(row) == "unactionable_domain"

    sensor = _label_row("intent__HassTurnOff", {"domain": ["sensor"]})
    assert gate_named_device_calls(sensor) == "unactionable_domain"


def test_gate_unactionable_domain_accepts_actionable_light():
    row = _label_row("intent__HassTurnOn", {"name": "Den Ceiling Lights", "domain": ["light"]})
    assert gate_named_device_calls(row) is None


def test_gate_light_color_rejects_non_home_assistant_color_names():
    row = _label_row("light__HassLightSet", {"name": "Den Ceiling Lights", "color": "warm white"})
    assert gate_named_device_calls(row) == "unknown_light_color"


def test_gate_light_color_accepts_home_assistant_color_name():
    row = _label_row("light__HassLightSet", {"name": "Den Ceiling Lights", "color": "red"})
    assert gate_named_device_calls(row) is None
