from __future__ import annotations

import copy
import json
import random
import re

import pytest
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators.context import system_prompt
from generators.tools import production_catalog
from generators.utterances import discrimination_utterance, finalize_training_utterance, vary_training_utterance
from generators.validation import (
    gate_areas,
    gate_clarification,
    gate_hidden_capability_refusal,
    gate_named_device_calls,
    gate_offered_tools,
    gate_on_off_polarity,
    gate_scripts,
    validate_serialized_row,
)

UTTERANCE = "turn on the den lights"
DEN_LIGHT = "Den Ceiling Lights"

HOME = {
    "areas": ["Den", "Living Room"],
    "entities": [
        {"name": DEN_LIGHT, "domain": "light", "capability": "lights", "capabilities": [],
         "area": "Den", "aliases": ["Den Lights"], "entity_id": "light.den_ceiling_lights"},
        {"name": "Living Room TV", "domain": "media_player", "capability": "media_players",
         "capabilities": ["on", "off", "mute", "volume"], "area": "Living Room", "aliases": [],
         "entity_id": "media_player.living_room_tv"},
        {"name": "Good Night", "domain": "script", "capability": None, "capabilities": [],
         "aliases": [], "entity_id": "script.good_night"},
    ],
}

TURN_ON_LIGHT = ("intent__HassTurnOn", {"name": DEN_LIGHT, "domain": ["light"]})


def _row(calls, *, metadata=None, trained=True, home=HOME):
    assistant = {
        "role": "assistant",
        "content": "",
        "train_on_turn": trained,
        "tool_calls": [
            {"id": f"call_{index}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for index, (name, arguments) in enumerate(calls)
        ],
    }
    return {
        "messages": [
            {"role": "system", "content": system_prompt(home, UTTERANCE), "train_on_turn": False},
            {"role": "user", "content": UTTERANCE, "train_on_turn": False},
            assistant,
        ],
        "tools": production_catalog(home),
        "metadata": metadata or {},
    }


def _serialized(row):
    return validate_serialized_row(row, token_budget=10**9, tokenizer_model="unused")


def test_g1_named_device_requires_exposed_canonical_name():
    assert gate_named_device_calls(_row([TURN_ON_LIGHT])) is None
    assert gate_named_device_calls(_row([("intent__HassTurnOn", {"name": "Den Lights", "domain": ["light"]})])) == "unknown_canonical_entity"
    assert _serialized(_row([TURN_ON_LIGHT])) is None


def test_g1_named_device_absent_from_context_is_rejected():
    row = _row([("intent__HassTurnOn", {"name": "Invented Lamp", "domain": ["light"]})])
    assert gate_named_device_calls(row) == "unknown_canonical_entity"
    assert _serialized(row) == "unknown_canonical_entity"


def test_g1_label_metadata_cannot_legitimise_an_unexposed_target():
    row = _row(
        [("intent__HassTurnOn", {"name": "Invented Lamp", "domain": ["light"]})],
        metadata={"expected_target_names": ["Invented Lamp"]},
    )
    assert gate_named_device_calls(row) == "unknown_canonical_entity"


def test_g1_untrained_history_turn_is_gated_like_trained():
    row = _row([("intent__HassTurnOn", {"name": "Invented Lamp", "domain": ["light"]})], trained=False)
    assert gate_named_device_calls(row) == "unknown_canonical_entity"
    assert _serialized(row) == "unknown_canonical_entity"


def test_g1_history_call_to_unexposed_name_before_retry_is_rejected():
    row = _row([TURN_ON_LIGHT])
    history = {
        "role": "assistant",
        "content": "",
        "train_on_turn": False,
        "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "intent__HassTurnOn",
                          "arguments": json.dumps({"name": "Bathroom Beth's Hood Fan", "domain": ["fan"]})}},
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": json.dumps({"error": {"code": "schema_mismatch"}}),
        "train_on_turn": False,
    }
    row["messages"] = [row["messages"][0], row["messages"][1], history, tool_result, row["messages"][2]]
    assert gate_named_device_calls(row) == "unknown_canonical_entity"
    assert _serialized(row) == "unknown_canonical_entity"


def test_g1_history_call_to_exposed_name_before_retry_passes():
    row = _row([TURN_ON_LIGHT])
    history = {
        "role": "assistant",
        "content": "",
        "train_on_turn": False,
        "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "intent__HassTurnOn",
                          "arguments": json.dumps({"name": DEN_LIGHT, "domain": ["light"]})}},
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": json.dumps({"error": {"code": "schema_mismatch"}}),
        "train_on_turn": False,
    }
    row["messages"] = [row["messages"][0], row["messages"][1], history, tool_result, row["messages"][2]]
    assert gate_named_device_calls(row) is None
    assert _serialized(row) is None


def test_g1_timer_without_device_name_is_not_a_violation():
    row = _row([("intent__HassTimerStatus", {})])
    assert gate_named_device_calls(row) is None
    assert _serialized(row) is None


def test_g1_script_exposed_in_offered_tools_passes():
    row = _row([("good_night", {})])
    assert gate_scripts(row) is None
    assert _serialized(row) is None


def test_g1_script_not_exposed_is_rejected():
    row = _row([("movie_night", {})])
    assert gate_scripts(row) == "unexposed_script"
    assert _serialized(row) is not None


def test_g1_area_visible_in_context_passes():
    row = _row([("intent__HassTurnOn", {"area": "Den", "domain": ["light"]})])
    assert gate_areas(row) is None
    assert _serialized(row) is None


def test_g1_area_absent_from_context_is_rejected():
    row = _row([("intent__HassTurnOff", {"area": "Garage", "domain": ["light"]})])
    assert gate_areas(row) == "unknown_area"
    assert _serialized(row) == "unknown_area"


def test_g4_offered_tool_passes():
    assert gate_offered_tools(_row([TURN_ON_LIGHT])) is None


def test_g4_tool_absent_from_catalog_is_rejected():
    row = _row([("intent__HassSetAlarm", {"name": DEN_LIGHT})])
    assert gate_offered_tools(row) == "unoffered_tool"
    assert _serialized(row) == "unoffered_tool"


CLARIFY_HOME = {
    "areas": ["Den", "Living Room", "Bedroom"],
    "entities": [
        {"name": "Den Ceiling Lights", "domain": "light", "capability": "lights", "capabilities": [],
         "area": "Den", "aliases": ["Den Lights"], "entity_id": "light.den_ceiling_lights"},
        {"name": "Den Lamp", "domain": "light", "capability": "lights", "capabilities": [],
         "area": "Den", "aliases": [], "entity_id": "light.den_lamp"},
        {"name": "Living Room TV", "domain": "media_player", "capability": "media_players",
         "capabilities": ["on", "off", "mute", "volume"], "area": "Living Room", "aliases": [],
         "entity_id": "media_player.living_room_tv"},
        {"name": "Living Room Speaker", "domain": "media_player", "capability": "media_players",
         "capabilities": ["on", "off", "volume"], "area": "Living Room", "aliases": [],
         "entity_id": "media_player.living_room_speaker"},
        {"name": "Bedroom TV", "domain": "media_player", "capability": "media_players",
         "capabilities": ["on", "off", "mute", "volume"], "area": "Bedroom", "aliases": [],
         "entity_id": "media_player.bedroom_tv"},
    ],
}


def _clarify_row(reply, utterance, home=CLARIFY_HOME):
    return {
        "messages": [
            {"role": "system", "content": system_prompt(home, utterance), "train_on_turn": False},
            {"role": "user", "content": utterance, "train_on_turn": False},
            {"role": "assistant", "content": reply, "train_on_turn": True},
        ],
        "tools": production_catalog(home),
        "metadata": {},
    }


def test_clarify_candidates_exposed_and_distinct_passes():
    row = _clarify_row("Did you mean the Living Room TV or the Bedroom TV?", "turn on the TV")
    assert gate_clarification(row) is None
    assert _serialized(row) is None


def test_clarify_candidate_not_exposed_is_rejected():
    row = _clarify_row("Did you mean the Living Room TV or the Garage TV?", "turn on the TV")
    assert gate_clarification(row) == "clarify_unexposed_candidate"
    assert _serialized(row) == "clarify_unexposed_candidate"


def test_clarify_same_entity_offered_twice_is_rejected():
    row = _clarify_row("Did you mean the Den Ceiling Lights or the Den Lights?", "turn on the den lights")
    assert gate_clarification(row) == "clarify_duplicate_candidate"
    assert _serialized(row) == "clarify_duplicate_candidate"


def test_clarify_distinct_entities_matching_the_reference_pass():
    row = _clarify_row("Did you mean the Den Ceiling Lights or the Den Lamp?", "turn on the den lights")
    assert gate_clarification(row) is None


def test_clarify_two_genuine_matches_for_generic_reference_pass():
    row = _clarify_row("Did you mean the Living Room TV or the Bedroom TV?", "turn on the TV")
    assert gate_clarification(row) is None


def test_clarify_tv_vs_speaker_for_living_room_tv_is_rejected():
    row = _clarify_row(
        "Did you mean the Living Room TV or the Living Room Speaker?", "turn on the living room TV"
    )
    assert gate_clarification(row) == "clarify_without_genuine_choice"
    assert _serialized(row) == "clarify_without_genuine_choice"


def test_clarify_single_candidate_is_rejected():
    row = _clarify_row("Did you mean the Living Room TV?", "turn on the TV")
    assert gate_clarification(row) == "clarify_without_genuine_choice"


@pytest.mark.parametrize("reply", [
    "Did you mean the Den Lights or the Den Lamp?",
    "Did you mean the Den Ceiling Lights or the Den Lamp?",
])
def test_clarify_canonical_or_alias_candidates_resolve_to_distinct_devices(reply):
    assert _serialized(_clarify_row(reply, "turn on the den lights")) is None


def test_clarify_metadata_cannot_expose_candidates():
    row = _clarify_row("Did you mean the Living Room TV or the Garage TV?", "turn on the TV")
    row["metadata"] = {"candidates": ["Living Room TV", "Garage TV"], "expected_target_names": ["Garage TV"]}
    assert _serialized(row) == "clarify_unexposed_candidate"


@pytest.mark.parametrize("utterance,reason", [
    ("turn on the living room projector", None),
    ("turn on the garage TV", "clarify_without_genuine_choice"),
    ("turn on the living room ceiling TV", "clarify_without_genuine_choice"),
])
def test_clarify_missing_reference_is_not_replaced_by_visible_words(utterance, reason):
    row = _clarify_row("Did you mean the Living Room TV or the Living Room Speaker?", utterance)
    assert _serialized(row) == reason


@pytest.mark.parametrize("reply", ["Which device did you mean?", "Did you mean the Living Room TV?"])
def test_clarify_without_two_offered_devices_is_rejected(reply):
    assert _serialized(_clarify_row(reply, "turn on the TV")) == "clarify_without_genuine_choice"


def test_clarify_untrained_turn_is_not_gated():
    row = _clarify_row("Did you mean the Missing TV?", "turn on the TV")
    row["messages"][-1]["train_on_turn"] = False
    assert _serialized(row) is None


def test_non_clarification_responses_are_not_device_choices():
    for reply in ("Done.", "Which area do you mean?", "There is no TV in the garage."):
        assert _serialized(_clarify_row(reply, "turn on the garage TV")) is None


def test_clarify_three_candidates_uses_generated_oxford_comma():
    from generators.rendering import _did_you_mean

    home = copy.deepcopy(CLARIFY_HOME)
    home["entities"].append({"name": "Den TV", "domain": "media_player", "area": "Den", "aliases": []})
    reply = _did_you_mean(["Living Room TV", "Bedroom TV", "Den TV"])
    assert _serialized(_clarify_row(reply, "turn on the TV", home)) is None


@pytest.mark.parametrize("second,reason", [
    ("Bedroom", None),
    ("Garage", "clarify_unexposed_candidate"),
    ("Den", "clarify_duplicate_candidate"),
])
def test_clarify_same_named_devices_resolve_by_offered_room(second, reason):
    home = copy.deepcopy(CLARIFY_HOME)
    home["entities"] = [
        {"name": "Lamp", "domain": "light", "area": area, "aliases": []}
        for area in ("Den", "Bedroom")
    ]
    reply = f"Which Lamp do you mean, the one in the Den or the {second}?"
    assert _serialized(_clarify_row(reply, "turn on the lamp", home)) == reason


def test_clarify_scripts_use_visible_tool_descriptions():
    home = copy.deepcopy(HOME)
    home["entities"].append({"name": "Movie Night", "domain": "script", "entity_id": "script.movie_night", "aliases": ["Cinema"]})
    row = _clarify_row("Did you mean the Good Night or the Cinema?", "run the routine", home)
    assert _serialized(row) is None
    row["messages"][-1]["content"] = "Did you mean the Movie Night or the Cinema?"
    assert _serialized(row) == "clarify_duplicate_candidate"
    row["messages"][-1]["content"] = "Did you mean the Good Night or the Missing Routine?"
    assert _serialized(row) == "clarify_unexposed_candidate"


def test_named_timer_is_not_a_device_target():
    assert _serialized(_row([("intent__HassStartTimer", {"name": "pasta", "minutes": 10})])) is None


def test_clarify_timers_require_visible_prior_starts():
    row = _clarify_row("Did you mean the pasta timer or the rice timer?", "pause the timer")
    history = _row([
        ("intent__HassStartTimer", {"name": "pasta", "minutes": 10}),
        ("intent__HassStartTimer", {"name": "rice", "minutes": 20}),
    ], trained=False)["messages"][-1]
    row["messages"].insert(1, history)
    assert _serialized(row) is None
    row["messages"].append(row["messages"].pop(1))
    assert _serialized(row) == "clarify_unexposed_candidate"


def test_exposed_device_with_unsupported_operation_is_rejected():
    row = _row([("media_player__HassSetVolume", {"name": DEN_LIGHT, "volume_level": 50})])
    assert _serialized(row) == "unsupported_operation"


def test_script_arguments_are_rejected():
    assert _serialized(_row([("good_night", {"name": DEN_LIGHT})])) == "script_tool_takes_no_arguments"


TURN_OFF_LIGHT = ("intent__HassTurnOff", {"name": DEN_LIGHT, "domain": ["light"]})
BLIND_CALLS = {"open": ("intent__HassTurnOn", {"name": "Living Room Blinds", "domain": ["cover"]}),
               "close": ("intent__HassTurnOff", {"name": "Living Room Blinds", "domain": ["cover"]})}
LOCK_CALLS = {"lock": ("intent__HassTurnOn", {"name": "Front Door Lock", "domain": ["lock"]}),
              "unlock": ("intent__HassTurnOff", {"name": "Front Door Lock", "domain": ["lock"]})}


def _polarity_row(utterance, calls, *, trained=True, home=HOME):
    row = _row(calls, trained=trained, home=home)
    row["messages"][1]["content"] = utterance
    return row


@pytest.mark.parametrize("utterance,call,reason", [
    ("turn on the den lights", TURN_OFF_LIGHT, "turn_off_answers_on_utterance"),
    ("turn off the den lights", TURN_ON_LIGHT, "turn_on_answers_off_utterance"),
    ("switch the den lights off", TURN_ON_LIGHT, "turn_on_answers_off_utterance"),
    ("switch on the den lights", TURN_OFF_LIGHT, "turn_off_answers_on_utterance"),
])
def test_polarity_contradicting_turn_on_off_is_rejected(utterance, call, reason):
    row = _polarity_row(utterance, [call])
    assert gate_on_off_polarity(row) == reason
    assert _serialized(row) == reason


@pytest.mark.parametrize("utterance,call", [
    ("turn on the den lights", TURN_ON_LIGHT),
    ("switch the den lights off", TURN_OFF_LIGHT),
])
def test_polarity_matching_turn_on_off_is_accepted(utterance, call):
    row = _polarity_row(utterance, [call])
    assert gate_on_off_polarity(row) is None
    assert _serialized(row) is None


@pytest.mark.parametrize("utterance,call,reason", [
    ("open the living room blinds", BLIND_CALLS["close"], "turn_off_answers_on_utterance"),
    ("raise the living room blinds", BLIND_CALLS["close"], "turn_off_answers_on_utterance"),
    ("close the living room blinds", BLIND_CALLS["open"], "turn_on_answers_off_utterance"),
    ("lower the living room blinds", BLIND_CALLS["open"], "turn_on_answers_off_utterance"),
])
def test_polarity_cover_open_close_contradiction_is_rejected(utterance, call, reason):
    assert gate_on_off_polarity(_polarity_row(utterance, [call])) == reason


@pytest.mark.parametrize("utterance,call", [
    ("open the living room blinds", BLIND_CALLS["open"]),
    ("raise the living room blinds", BLIND_CALLS["open"]),
    ("close the living room blinds", BLIND_CALLS["close"]),
    ("shut the living room blinds", BLIND_CALLS["close"]),
])
def test_polarity_cover_open_close_matching_is_accepted(utterance, call):
    assert gate_on_off_polarity(_polarity_row(utterance, [call])) is None


@pytest.mark.parametrize("utterance,call,reason", [
    ("unlock the front door", LOCK_CALLS["lock"], "turn_on_answers_off_utterance"),
    ("lock the front door", LOCK_CALLS["unlock"], "turn_off_answers_on_utterance"),
])
def test_polarity_lock_follows_generator_mapping(utterance, call, reason):
    assert gate_on_off_polarity(_polarity_row(utterance, [call])) == reason


@pytest.mark.parametrize("utterance,call", [
    ("lock the front door", LOCK_CALLS["lock"]),
    ("unlock the front door", LOCK_CALLS["unlock"]),
])
def test_polarity_lock_matching_mapping_is_accepted(utterance, call):
    assert gate_on_off_polarity(_polarity_row(utterance, [call])) is None


@pytest.mark.parametrize("call", [TURN_ON_LIGHT, TURN_OFF_LIGHT])
def test_polarity_toggle_is_not_rejected(call):
    assert gate_on_off_polarity(_polarity_row("toggle the den lights", [call])) is None


def test_polarity_multi_action_with_both_operations_is_skipped():
    row = _polarity_row(
        "turn off the den lights and turn on the living room TV",
        [TURN_OFF_LIGHT, ("intent__HassTurnOn", {"name": "Living Room TV", "domain": ["media_player"]})],
    )
    assert gate_on_off_polarity(row) is None


def test_polarity_uses_nearest_preceding_user_turn():
    row = _polarity_row("turn off the den lights", [TURN_OFF_LIGHT])
    row["messages"].insert(2, {"role": "assistant", "content": "Done.", "train_on_turn": False})
    row["messages"].insert(3, {"role": "user", "content": "turn on the den lights", "train_on_turn": False})
    assert gate_on_off_polarity(row) == "turn_off_answers_on_utterance"


def test_polarity_untrained_turn_is_not_gated():
    row = _polarity_row("turn on the den lights", [TURN_OFF_LIGHT], trained=False)
    assert gate_on_off_polarity(row) is None


def _refusal_row(reply, utterance, *, home=CLARIFY_HOME, removed_tools=()):
    return {
        "messages": [
            {"role": "system", "content": system_prompt(home, utterance), "train_on_turn": False},
            {"role": "user", "content": utterance, "train_on_turn": False},
            {"role": "assistant", "content": reply, "train_on_turn": True},
        ],
        "tools": production_catalog(home, removed_tools=list(removed_tools)),
        "metadata": {},
    }


def test_hidden_capability_refusal_for_exposed_capable_device_is_rejected():
    row = _refusal_row("Den Ceiling Lights does not support that.", "turn on the den lights")
    assert gate_hidden_capability_refusal(row) == "hidden_capability_refusal"
    assert _serialized(row) == "hidden_capability_refusal"


def test_hidden_capability_refusal_for_exposed_speaker_is_rejected():
    row = _refusal_row("Living Room Speaker does not support that.", "turn on the living room speaker")
    assert gate_hidden_capability_refusal(row) == "hidden_capability_refusal"


def test_refusal_for_device_not_exposed_is_justified():
    row = _refusal_row("Garage Lamp does not support that.", "turn on the garage lamp")
    assert gate_hidden_capability_refusal(row) is None


def test_refusal_when_domain_cannot_do_the_operation_is_justified():
    row = _refusal_row("Den Ceiling Lights does not support that.", "pause the den lights")
    assert gate_hidden_capability_refusal(row) is None


def test_refusal_when_required_tool_is_withheld_is_justified():
    row = _refusal_row(
        "Living Room TV does not support that.",
        "turn on the living room TV",
        removed_tools=["intent__HassTurnOn"],
    )
    assert "intent__HassTurnOn" not in {tool["function"]["name"] for tool in row["tools"]}
    assert gate_hidden_capability_refusal(row) is None


def test_refusal_for_area_without_that_device_type_is_justified():
    row = _refusal_row("The Bedroom has no light available.", "turn on the bedroom lights")
    assert gate_hidden_capability_refusal(row) is None


def test_hidden_capability_refusal_is_not_gated_on_untrained_turn():
    row = _refusal_row("Den Ceiling Lights does not support that.", "turn on the den lights")
    row["messages"][-1]["train_on_turn"] = False
    assert gate_hidden_capability_refusal(row) is None


def test_audit_corpus_counts_each_gate_over_good_and_bad_rows(tmp_path):
    from generators.audit import audit_corpus

    good = _row([TURN_ON_LIGHT], metadata={"candidate_id": "good", "family": "direct"})
    unknown_device = _row(
        [("intent__HassTurnOn", {"name": "Invented Lamp", "domain": ["light"]})],
        metadata={"candidate_id": "bad_device", "family": "invented"},
    )
    unoffered_tool = _row([("intent__HassSetAlarm", {"name": DEN_LIGHT})], metadata={"family": "alarm"})
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in (good, unknown_device, unoffered_tool)) + "\n")

    report = audit_corpus(corpus)

    assert report["rows_scanned"] == 3
    assert report["rows_rejected"] == 2
    assert report["gates"]["gate_named_device_calls"] == {
        "rejected": 2,
        "reasons": {"unknown_canonical_entity": 1, "unoffered_tool": 1},
        "examples": ["bad_device", 3],
    }
    assert report["gates"]["gate_offered_tools"] == {
        "rejected": 1, "reasons": {"unoffered_tool": 1}, "examples": [3],
    }
    assert report["gates"]["gate_scripts"]["rejected"] == 0
    assert report["rejected_by_metadata"] == {"family": {"alarm": 1, "invented": 1}}
    assert report["label_conflicts"]["groups"] == 1
    assert report["label_conflicts"]["rows"] == 3
    assert audit_corpus(corpus, limit=1)["rows_scanned"] == 1


@pytest.mark.parametrize("utterance,reply,names,domain,area,reason", [
    (
        "Could you turn on the Laundry Room tv?",
        "Did you mean the SaySo Satellite or the Laundry Room Nest Mini?",
        ("SaySo Satellite", "Laundry Room Nest Mini"), "media_player", "Laundry Room",
        "clarify_without_genuine_choice",
    ),
    (
        "Switch on the Kitchen tv",
        "Did you mean the Kitchen radio or the Kitchen Sharp TV?",
        ("Kitchen radio", "Kitchen Sharp TV"), "media_player", "Kitchen",
        "clarify_without_genuine_choice",
    ),
    (
        "Could you set the Living Room tv volume to 35 percent?",
        "Did you mean the Sally's Smart Speaker or the Living Room Speaker?",
        ("Sally's Smart Speaker", "Living Room Speaker"), "media_player", "Living Room",
        "clarify_without_genuine_choice",
    ),
    (
        "set the kitchen fan speed to 32 percent",
        "Did you mean the Kitchen Range Hood Fan or the Kitchen Bond Fan?",
        ("Kitchen Range Hood Fan", "Kitchen Bond Fan"), "fan", "Kitchen", None,
    ),
    (
        "what is the status of the entryway light",
        "Did you mean the Entryway Ceiling Lights or the Entryway arc lamp?",
        ("Entryway Ceiling Lights", "Entryway arc lamp"), "light", "Entryway", None,
    ),
    (
        "Decrease the temperature of the Garage thermostat to 68",
        "Did you mean the Garage Ductless AC or the Garage Heating?",
        ("Garage Ductless AC", "Garage Heating"), "climate", "Garage", None,
    ),
    (
        "please skip to the next track on the living room tv.",
        "Did you mean the Living Room Smart TV or the Living Room Smart TV 2?",
        ("Living Room Smart TV", "Living Room Smart TV 2"), "media_player", "Living Room", None,
    ),
    (
        "could you turn the bathroom van on?",
        "Did you mean the Bathroom Hood Fan or the Bathroom Xiaomi Fan?",
        ("Bathroom Hood Fan", "Bathroom Xiaomi Fan"), "fan", "Bathroom", None,
    ),
    (
        "Switch the garaj outlet off",
        "Did you mean the Garage Wall Outlet or the Garage Printer Plug?",
        ("Garage Wall Outlet", "Garage Printer Plug"), "switch", "Garage", None,
    ),
])
def test_clarify_v12_type_words(utterance, reply, names, domain, area, reason):
    home = {
        "areas": [area],
        "entities": [{"name": name, "domain": domain, "area": area, "aliases": []} for name in names],
    }
    row = _clarify_row(reply, utterance, home)
    assert gate_clarification(row) == reason
    assert _serialized(row) == reason


@pytest.mark.parametrize("word", ["tv", "t v", "t.v.", "television", "TVs", "televisions"])
def test_clarify_tv_spellings_require_two_visible_tvs(word):
    row = _clarify_row("Did you mean the Living Room TV or the Living Room Speaker?", f"turn on the {word}")
    assert _serialized(row) == "clarify_without_genuine_choice"
    row["messages"][-1]["content"] = "Did you mean the Living Room TV or the Bedroom TV?"
    assert _serialized(row) is None


def test_clarify_tv_alias_counts_but_tv_inside_a_word_does_not():
    home = copy.deepcopy(CLARIFY_HOME)
    home["entities"][3]["aliases"] = ["Living Room Television"]
    reply = "Did you mean the Living Room TV or the Living Room Speaker?"
    assert _serialized(_clarify_row(reply, "turn on the living room tv", home)) is None
    home["entities"][3]["aliases"] = ["Living Room TVstand"]
    assert _serialized(_clarify_row(reply, "turn on the living room tv", home)) == "clarify_without_genuine_choice"


def test_clarify_visible_area_limits_genuine_matches():
    row = _clarify_row("Did you mean the Living Room TV or the Bedroom TV?", "turn on the living room tv")
    assert _serialized(row) == "clarify_without_genuine_choice"


@pytest.mark.parametrize("reply,reason", [
    ("Did you mean the Den Ceiling Lights or the Den Lamp?", None),
    ("Did you mean the Den Ceiling Lights or the Garage Lamp?", "clarify_unexposed_candidate"),
    ("Did you mean the Den Ceiling Lights or the Den Lights?", "clarify_duplicate_candidate"),
])
@pytest.mark.parametrize("utterance", ["turn on the den lite", "switch on the den lite"])
def test_clarify_stt_noise_skips_only_genuine_choice(reply, reason, utterance):
    assert _serialized(_clarify_row(reply, utterance)) == reason


def test_clarify_candidate_case_and_the_prefix_resolve():
    row = _clarify_row("Did you mean THE DEN CEILING LIGHTS or den lamp?", "turn on the den lights")
    assert _serialized(row) is None


def test_clarify_uses_the_user_turn_it_answers():
    row = _clarify_row("Did you mean the Living Room TV or the Bedroom TV?", "turn on the den lights")
    row["messages"].insert(2, {"role": "assistant", "content": "Done.", "train_on_turn": False})
    row["messages"].insert(3, {"role": "user", "content": "turn on the tv", "train_on_turn": False})
    assert _serialized(row) is None


def _place_entity(name, domain, area, capability):
    return {"name": name, "domain": domain, "capability": capability, "capabilities": [],
            "area": area, "aliases": [], "entity_id": f"{domain}.{name.casefold().replace(' ', '_')}"}


POLARITY_HOME = {
    "areas": ["Main Floor", "Upstairs", "Garage", "Guest Room", "Office", "Living Room", "Master Bedroom", "Entryway"],
    "entities": [
        _place_entity("Office Downlight", "light", "Office", "lights"),
        _place_entity("Guest Room Ceiling Light", "light", "Guest Room", "lights"),
        _place_entity("Hilary's Downlights", "light", "Upstairs", "lights"),
        _place_entity("Garage Clip Fan", "fan", "Garage", "fans"),
        _place_entity("Living Room Backlight Lights", "light", "Living Room", "lights"),
        _place_entity("Streaming TVs", "media_player", "Living Room", "media_players"),
        _place_entity("Ledvance Lights", "light", "Master Bedroom", "lights"),
        _place_entity("Heater Outlets", "switch", "Garage", "switches"),
        _place_entity("Entryway All Off", "scene", "Entryway", "scenes"),
    ],
}

ON_CALL = ("intent__HassTurnOn", {"name": "Office Downlight", "domain": ["light"]})
OFF_CALL = ("intent__HassTurnOff", {"name": "Office Downlight", "domain": ["light"]})


@pytest.mark.parametrize("utterance,calls", [
    ("Turn the outlets on Main Floor off",
     [("intent__HassTurnOff", {"domain": ["switch"], "floor": "Main Floor"})]),
    ("koda Turn the thermostats on Upstairs off",
     [("intent__HassTurnOff", {"domain": ["climate"], "floor": "Upstairs"})]),
    ("Can you turn thermostats on Main Floor off for me?",
     [("intent__HassTurnOff", {"domain": ["climate"], "floor": "Main Floor"})]),
    ("turn hilary's downlights to on and turn off garage clip fan",
     [("intent__HassTurnOn", {"name": "Hilary's Downlights", "domain": ["light"]}),
      ("intent__HassTurnOff", {"name": "Garage Clip Fan", "domain": ["fan"]})]),
    ("switch the office downlight off and the guest room ceiling light on",
     [("intent__HassTurnOff", {"name": "Office Downlight", "domain": ["light"]}),
      ("intent__HassTurnOn", {"name": "Guest Room Ceiling Light", "domain": ["light"]})]),
    ("please change to entryway all off",
     [("intent__HassTurnOn", {"domain": ["scene"], "name": "Entryway All Off"})]),
])
def test_polarity_accepts_on_as_preposition_and_mixed_or_scene_turns(utterance, calls):
    assert gate_on_off_polarity(_polarity_row(utterance, calls, home=POLARITY_HOME)) is None


@pytest.mark.parametrize("utterance,call", [
    ("switch on the living room backlight lights",
     ("intent__HassTurnOff", {"name": "Living Room Backlight Lights", "domain": ["light"]})),
    ("Can you turn on Streaming TVs in my Living Room for me?",
     ("intent__HassTurnOff", {"name": "Streaming TVs", "domain": ["media_player"]})),
    ("can you turn on the ledvance lights in the master bedroom for me?",
     ("intent__HassTurnOff", {"name": "Ledvance Lights", "domain": ["light"]})),
    ("please switch on the heater outlets in the garage",
     ("intent__HassTurnOff", {"name": "Heater Outlets", "domain": ["switch"]})),
])
def test_polarity_real_v12_turn_on_labelled_off_stays_rejected(utterance, call):
    row = _polarity_row(utterance, [call], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) == "turn_off_answers_on_utterance"
    assert _serialized(row) == "turn_off_answers_on_utterance"


def test_polarity_on_before_place_without_trailing_off_still_reads_as_on():
    row = _polarity_row("turn on the living room lights", [("intent__HassTurnOff", {"name": "Streaming TVs", "domain": ["light"]})], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) == "turn_off_answers_on_utterance"


def test_polarity_visible_area_read_from_the_prompt_is_a_preposition():
    row = _polarity_row("Turn the outlets on Garage off", [("intent__HassTurnOff", {"domain": ["switch"], "area": "Garage"})], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) is None


def test_polarity_both_operations_in_one_trained_turn_is_skipped():
    row = _polarity_row(
        "switch the office downlight off and the guest room ceiling light on",
        [OFF_CALL, ("intent__HassTurnOn", {"name": "Guest Room Ceiling Light", "domain": ["light"]})],
        home=POLARITY_HOME,
    )
    assert gate_on_off_polarity(row) is None


def test_polarity_scene_target_is_skipped_even_with_a_wrong_operation():
    row = _polarity_row("turn on the entryway all off", [("intent__HassTurnOff", {"domain": ["scene"], "name": "Entryway All Off"})], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) is None


def test_polarity_target_name_containing_the_polarity_word_is_skipped():
    row = _polarity_row("turn off the office downlight", [("intent__HassTurnOn", {"name": "Office Downlight Off", "domain": ["light"]})], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) is None


KITCHEN_HOME = {
    "areas": ["Kitchen", "Entryway", "Garage"],
    "entities": [
        _place_entity("Blender Plug", "switch", "Kitchen", "switches"),
        _place_entity("Entryway Hue Light", "light", "Entryway", "lights"),
    ],
}


@pytest.mark.parametrize("call_name,seed,polarity,opposite", [
    ("HassTurnOn", "turn the blender outlets in the kitchen on", "on", "off"),
    ("HassTurnOff", "turn the blender outlets in the kitchen off", "off", "on"),
    ("HassTurnOff", "switch off the blender outlets in the kitchen", "off", "on"),
    ("HassTurnOn", "switch the blender outlets in the kitchen on", "on", "off"),
])
def test_discrimination_plural_phrasing_follows_the_operation(call_name, seed, polarity, opposite):
    description = "the blender outlets in the kitchen"
    for index in range(40):
        utterance = finalize_training_utterance(
            discrimination_utterance(call_name, seed, description), random.Random(index)
        )
        tokens = re.findall(r"[a-z]+", utterance.lower())
        assert polarity in tokens and opposite not in tokens, utterance
        row = _polarity_row(utterance, [(f"intent__{call_name}", {"name": "Blender Plug", "domain": ["switch"]})], home=KITCHEN_HOME)
        assert gate_on_off_polarity(row) is None, utterance


def test_discrimination_utterance_keeps_a_matching_operation_verb():
    assert discrimination_utterance("intent__HassTurnOff", "close the garage door", "the garage door") == "close the garage door"
    assert discrimination_utterance("intent__HassTurnOn", "open the garage door", "the garage door") == "open the garage door"


@pytest.mark.parametrize("utterance,call,reason", [
    ("turn the wall fans in my living room on", ("intent__HassTurnOff", {"name": "Living Room Wall Fans", "domain": ["fan"]}), "turn_off_answers_on_utterance"),
    ("switch the blender outlets in the kitchen on please", ("intent__HassTurnOff", {"name": "Blender Plug", "domain": ["switch"]}), "turn_off_answers_on_utterance"),
    ("turn the hue lights in my entryway off for me", ("intent__HassTurnOn", {"name": "Entryway Hue Light", "domain": ["light"]}), "turn_on_answers_off_utterance"),
    ("turn the wall fans in my living room on.", ("intent__HassTurnOff", {"name": "Living Room Wall Fans", "domain": ["fan"]}), "turn_off_answers_on_utterance"),
])
def test_polarity_trailing_particle_contradiction_is_rejected(utterance, call, reason):
    assert gate_on_off_polarity(_polarity_row(utterance, [call], home=POLARITY_HOME)) == reason


@pytest.mark.parametrize("utterance,call", [
    ("turn the wall fans in my living room on", ("intent__HassTurnOn", {"name": "Living Room Wall Fans", "domain": ["fan"]})),
    ("switch the blender outlets in the kitchen on please", ("intent__HassTurnOn", {"name": "Blender Plug", "domain": ["switch"]})),
    ("turn the hue lights in my entryway off for me", ("intent__HassTurnOff", {"name": "Entryway Hue Light", "domain": ["light"]})),
    ("turn off the den lights and leave the fan on", ("intent__HassTurnOff", {"name": DEN_LIGHT, "domain": ["light"]})),
])
def test_polarity_trailing_particle_matching_is_accepted(utterance, call):
    assert gate_on_off_polarity(_polarity_row(utterance, [call], home=POLARITY_HOME)) is None


def test_polarity_trailing_particle_does_not_cross_a_clause_break():
    row = _polarity_row("turn off the den lights, keep the hall lights on", [TURN_OFF_LIGHT], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) is None
    row = _polarity_row("turn off the den lights and keep the hall lights on", [TURN_ON_LIGHT], home=POLARITY_HOME)
    assert gate_on_off_polarity(row) == "turn_on_answers_off_utterance"


def _ambiguity_home(names, *, aliases=None, device_class="speaker"):
    from generators.homes import make_entity

    return {
        "home_id": "ambiguity_regression", "size": len(names),
        "areas": ["Kitchen"], "area_floors": {"Kitchen": "Main Floor"},
        "sayso_entity_area": "Kitchen",
        "entities": [make_entity(
            name=name, capability="media_players", area="Kitchen", floor="Main Floor",
            rng=random.Random(index), aliases=(aliases or {}).get(name, []),
            device_class=device_class, features=("on", "off", "volume", "mute", "pause", "next"),
        ) for index, name in enumerate(names)],
    }


def _generated_ambiguity_row(monkeypatch, home, family, operation="turn_on", *, real_path=None):
    from generators.config import GeneratorConfig
    from generators.deduplication import DuplicateTracker
    from generators.row_generation import generate_row
    from generators.scenarios import core

    monkeypatch.setattr(core, "generate_home", lambda *args, **kwargs: copy.deepcopy(home))
    slot = {"index": 0, "family": family, "robustness": "ambiguity",
            "capability": home["entities"][0]["capability"], "operation": operation,
            "tier": 1, "home_size": len(home["entities"])}
    config = GeneratorConfig(count=1, seed=17, real_home_path=real_path,
                             real_home_rate=1.0 if real_path else 0.0)
    row, reason = generate_row(slot, config, random.Random(17), excluded=set(),
                               dup_tracker=DuplicateTracker())
    assert reason is None
    assert row is not None
    assert gate_clarification(row) is None
    return row


@pytest.mark.parametrize("family", ["clarify", "follow_up"])
@pytest.mark.parametrize("operation", ["turn_on", "volume_set", "query_state"])
@pytest.mark.parametrize("names,count", [
    (["Kitchen Sharp TV", "Kitchen Radio"], 1),
    (["Kitchen Sharp TV", "Kitchen Sony Television"], 2),
    (["Kitchen Bookshelf Speakers", "Kitchen Echo Show"], 0),
])
def test_generated_media_ambiguity_uses_visible_tv_matches(monkeypatch, family, operation, names, count):
    home = _ambiguity_home(names, device_class="tv" if count == 0 else "speaker")
    row = _generated_ambiguity_row(monkeypatch, home, family, operation)
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    assert re.search(r"\btv\b", user, re.I)
    clarifications = [m for m in row["messages"] if m["role"] == "assistant" and "Did you mean" in m.get("content", "")]
    calls = [c["function"] for m in row["messages"] for c in m.get("tool_calls", []) if m.get("train_on_turn")]
    if count == 2:
        assert len(clarifications) == 1
        assert all(name in clarifications[0]["content"] for name in names)
        assert row["metadata"]["family"] == family
        assert bool(calls) == (family == "follow_up")
    elif count == 1:
        assert not clarifications
        assert len(calls) == 1
        assert json.loads(calls[0]["arguments"])["name"] == names[0]
        assert row["metadata"]["family"] == ("status" if operation == "query_state" else "ordinary")
    else:
        assert not clarifications and not calls
        assert row["metadata"]["family"] == "absence"
        assert "tv" in row["messages"][-1]["content"].casefold()
    assert home["entities"] == _ambiguity_home(names, device_class="tv" if count == 0 else "speaker")["entities"]


@pytest.mark.parametrize("alias,count", [("Kitchen Television", 2), ("Kitchen TVstand", 1)])
@pytest.mark.parametrize("family", ["clarify", "follow_up"])
def test_generated_media_ambiguity_uses_visible_alias_words(monkeypatch, family, alias, count):
    names = ["Kitchen Sharp TV", "Kitchen Display"]
    home = _ambiguity_home(names, aliases={names[1]: [alias]})
    row = _generated_ambiguity_row(monkeypatch, home, family)
    clarification = [m for m in row["messages"] if "Did you mean" in m.get("content", "")]
    assert len(clarification) == (1 if count == 2 else 0)
    if count == 1:
        calls = [c["function"] for m in row["messages"] for c in m.get("tool_calls", [])]
        assert json.loads(calls[0]["arguments"])["name"] == names[0]


@pytest.mark.parametrize("family", ["clarify", "follow_up"])
def test_real_home_clarification_uses_rendered_entities(monkeypatch, tmp_path, family):
    from generators.homes import make_entity

    home = _ambiguity_home([])
    home["areas"].append("Hallway")
    home["area_floors"]["Hallway"] = "Main Floor"
    home["sayso_entity_area"] = "Hallway"
    home["entities"] = [make_entity(
        name=name, capability="switches", area=area, floor="Main Floor", rng=random.Random(index),
    ) for index, (name, area) in enumerate([
        ("Hallway Router Plug", "Hallway"), ("Kitchen Coffee Outlet", "Kitchen"),
        ("Hallway Tuya Outlet", "Hallway"),
    ])]
    home["size"] = len(home["entities"])
    path = tmp_path / "home.json"
    path.write_text(json.dumps(home))
    operation = "query_state" if family == "clarify" else "turn_on"
    row = _generated_ambiguity_row(monkeypatch, home, family, operation, real_path=path)
    assert row["metadata"]["real_home"]
    question = next(m["content"] for m in row["messages"] if "Did you mean" in m.get("content", ""))
    assert question == "Did you mean the Hallway Router Plug or the Hallway Tuya Outlet?"


def test_media_ambiguity_deduplicates_candidates():
    from generators.gold import gold_from_scenario

    home = _ambiguity_home(["Kitchen Sharp TV", "Kitchen Sony TV"])
    home["entities"].insert(1, copy.deepcopy(home["entities"][0]))
    expected = gold_from_scenario({"home": home, "capability": "media_players", "operation": "turn_on",
                                   "robustness": "ambiguity", "request_intent": {"name": "tv"}}, random.Random(0))
    assert expected["candidates"] == ["Kitchen Sharp TV", "Kitchen Sony TV"]


def test_prepare_media_tv_ambiguity_seeds_two_visible_tvs():
    from generators.homes import prepare_media_tv_ambiguity, _visible_tv_entities

    home = {
        "home_id": "slot_home",
        "areas": ["Kitchen"],
        "area_floors": {"Kitchen": "Main Floor"},
        "sayso_entity_area": "Kitchen",
        "entities": [],
    }
    prepare_media_tv_ambiguity(home, "turn_on", "Kitchen", random.Random(0), 0)
    assert len(_visible_tv_entities(home, "Kitchen")) >= 2


def test_timer_builders_accept_area_and_never_carry_both_scopes():
    from generators.tools import build_pause_timer, build_timer_status, build_unpause_timer

    for builder, name in ((build_pause_timer, "HassPauseTimer"), (build_unpause_timer, "HassUnpauseTimer"),
                          (build_timer_status, "HassTimerStatus")):
        assert builder(area="Den") == {"name": name, "arguments": {"area": "Den"}}
        assert builder(name="Tea") == {"name": name, "arguments": {"name": "Tea"}}
        assert builder() == {"name": name, "arguments": {}}
        assert builder(name="Tea", area="Den") == {"name": name, "arguments": {"name": "Tea"}}


def test_cancel_all_timers_is_sometimes_global_and_sometimes_area_scoped():
    from generators.tools import build_call_for_operation

    scopes = set()
    for seed in range(200):
        call = build_call_for_operation(None, "timers", "cancel_all", random.Random(seed), area="Den")
        assert call["name"] == "HassCancelAllTimers"
        scopes.add(call["arguments"].get("area"))
    assert scopes == {None, "Den"}


def test_timer_calls_pass_visible_area_only_when_unnamed():
    from generators.tools import build_call_for_operation

    for operation in ("pause", "unpause", "status"):
        areas = [
            build_call_for_operation(None, "timers", operation, random.Random(seed), area="Den")["arguments"]
            for seed in range(300)
        ]
        with_area = [arguments for arguments in areas if arguments.get("area")]
        assert 0.2 <= len(with_area) / len(areas) <= 0.4
        assert all(set(arguments) <= {"area"} and arguments.get("area", "Den") == "Den" for arguments in areas)


def test_area_status_call_renders_area_phrasing_and_validates():
    from generators.ohf import render_call

    call = {"name": "HassTimerStatus", "arguments": {"area": "Den"}}
    phrases = {render_call(call, "", seed=seed) for seed in range(20)}
    assert phrases and all(phrase and "Den" in phrase and "timer" in phrase for phrase in phrases)
    assert _serialized(_row([("intent__HassTimerStatus", {"area": "Den"})])) is None


def test_hold_and_freeze_are_timer_pause_requests_only():
    timer = {
        finalize_training_utterance("pause the kitchen timer", random.Random(seed), upper=False)
        for seed in range(60)
    }
    verbs = {verb for text in timer for verb in re.findall(r"\b(hold|freeze|pause)\b", text)}
    assert {"hold", "freeze", "pause"} <= verbs
    assert all(re.search(r"\b(hold|freeze|pause) (the|my) kitchen timer", text) for text in timer)
    for text in timer:
        row = _row([("intent__HassPauseTimer", {"area": "Den"})])
        row["messages"][1]["content"] = text
        assert _serialized(row) is None
    media = {
        finalize_training_utterance("pause the living room tv", random.Random(seed), upper=False)
        for seed in range(60)
    }
    assert all(text.startswith(("pause ", "can you pause ", "could you pause ", "please pause ")) for text in media)



def test_turn_on_script_keeps_the_script_tool_label():
    row = _polarity_row("turn on the good night", [("good_night", {})])
    assert gate_on_off_polarity(row) is None
    assert _serialized(row) is None


def test_switch_on_script_passes_serialization():
    assert _serialized(_polarity_row("switch on the good night", [("good_night", {})])) is None


def test_turn_on_phrasing_reaches_plain_script_rows():
    turned_on = [
        vary_training_utterance("run Conservatory Away Script", random.Random(seed))
        for seed in range(200)
    ]
    matched = [text for text in turned_on if text.startswith(("turn on ", "switch on "))]
    assert 20 <= len(matched) <= 70, len(matched)


def test_turn_on_phrasing_reaches_routine_phrasings():
    matched = [
        vary_training_utterance("run my Away Mode routine", random.Random(seed))
        for seed in range(200)
    ]
    assert any(text.startswith(("turn on ", "switch on ")) for text in matched)


@pytest.mark.parametrize("target", ["Kitchen Lights Off Routine", "Front Door On Alarm Script"])
def test_turn_on_phrasing_skips_script_names_with_polarity_words(target):
    for seed in range(200):
        text = vary_training_utterance(f"run {target}", random.Random(seed))
        assert not text.startswith(("turn on ", "switch on ")), text


def test_run_scripts_keep_the_run_start_activate_variants():
    seen = set()
    for seed in range(200):
        text = vary_training_utterance("run Movie Night Script", random.Random(seed))
        seen.update(re.findall(r"\b(run|start|activate)(?: the)? movie night script\b", text.casefold()))
    assert {"run", "start", "activate"} <= seen
