from __future__ import annotations

import random

import pytest

from generators.gold import expected_unexposed_domain, gold_from_scenario


def _entity(
    name,
    capability,
    domain,
    area,
    *,
    aliases=(),
    features=(),
    device_class=None,
    floor="Main Floor",
    state="off",
):
    return {
        "entity_id": f"{domain}.{name.casefold().replace(' ', '_')}",
        "name": name,
        "aliases": list(aliases),
        "domain": domain,
        "capability": capability,
        "device_class": device_class,
        "area": area,
        "floor": floor,
        "state": state,
        "capabilities": list(features),
    }


def _scenario(
    home_area,
    entities,
    capability,
    operation,
    *,
    target=None,
    targets=None,
    targeting="individual",
    area=None,
    robustness=None,
    intent=None,
):
    scenario = {
        "home": {"sayso_entity_area": home_area, "entities": list(entities)},
        "capability": capability,
        "operation": operation,
        "targeting": targeting,
    }
    if target is not None:
        scenario["target_entity"] = entities[target]
    if targets is not None:
        scenario["target_entities"] = [entities[index] for index in targets]
    if area is not None:
        scenario["area"] = area
    if robustness is not None:
        scenario["robustness"] = robustness
    if intent is not None:
        scenario["request_intent"] = intent
    return scenario


CASES = [
    pytest.param(
        _scenario(
            "Bedroom",
            [_entity("Bedroom Lamp", "lights", "light", "Bedroom", floor="Upstairs")],
            "lights", "turn_on", target=0,
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOn", "arguments": {"name": "Bedroom Lamp", "domain": ["light"]}},
        ]},
        id="rename-lamp-turn-on-new-name",
    ),
    pytest.param(
        _scenario(
            "Bedroom",
            [_entity("Bedroom Lamp", "lights", "light", "Bedroom", floor="Upstairs")],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "lamp 2"},
        ),
        {"kind": "no_action", "response": "device_absent", "calls": [],
         "unavailable": {"area": "bedroom", "type": "lamp 2"}},
        id="rename-old-name-request-is-absent",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [_entity("Living Room TV", "media_players", "media_player", "Living Room",
                     device_class="tv", features=("on", "off", "pause", "volume", "mute"))],
            "media_players", "pause", robustness="ambiguity", intent={"name": "tv"},
        ),
        {"kind": "action", "calls": [
            {"name": "HassMediaPause", "arguments": {
                "name": "Living Room TV", "domain": ["media_player"], "device_class": ["tv"]}},
        ]},
        id="rename-tv-pause-ambiguity-uses-new-name",
    ),
    pytest.param(
        _scenario(
            "Office",
            [_entity("Office Fan", "fans", "fan", "Office", features=("on", "off", "percentage"))],
            "fans", "turn_off", target=0,
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"name": "Office Fan", "domain": ["fan"]}},
        ]},
        id="rename-fan-turn-off-new-name",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Coffee Maker Plug", "switches", "switch", "Kitchen",
                     device_class="outlet", state="on")],
            "switches", "query_state", target=0,
        ),
        {"kind": "status",
         "calls": [{"name": "GetLiveContext",
                    "arguments": {"name": "Coffee Maker Plug", "domain": "switch"}}],
         "state": "on"},
        id="rename-switch-status-new-name",
    ),
    pytest.param(
        _scenario(
            "Entryway",
            [_entity("Front Door", "locks", "lock", "Entryway")],
            "locks", "unlock", target=0,
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"name": "Front Door"}},
        ]},
        id="rename-lock-unlock-new-name",
    ),
    pytest.param(
        _scenario(
            "Bedroom",
            [_entity("Bedroom Lamp", "lights", "light", "Bedroom", aliases=("Lamp 2",),
                     floor="Upstairs")],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "lamp 2"},
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOn", "arguments": {"name": "Bedroom Lamp", "domain": ["light"]}},
        ]},
        id="rename-kept-old-name-alias-resolves-canonical",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen", aliases=("Overhead",))],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "overhead"},
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOn", "arguments": {"name": "Kitchen Ceiling Light", "domain": ["light"]}},
        ]},
        id="alias-overhead-resolves-canonical-light",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Ceiling Light", "lights", "light", "Kitchen", aliases=("Overhead",)),
                _entity("Kitchen Counter Light", "lights", "light", "Kitchen"),
            ],
            "lights", "turn_off", robustness="ambiguity", intent={"name": "counter"},
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"name": "Kitchen Counter Light", "domain": ["light"]}},
        ]},
        id="alias-does-not-capture-similar-named-light",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Ceiling Light", "lights", "light", "Kitchen", aliases=("Main Light",)),
                _entity("Kitchen Island Light", "lights", "light", "Kitchen", aliases=("Main Light",)),
            ],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "main light"},
        ),
        {"kind": "no_action", "response": "clarify", "calls": [],
         "candidates": ["Kitchen Ceiling Light", "Kitchen Island Light"]},
        id="alias-shared-by-two-lights-clarifies",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Office Desk Lamp", "lights", "light", "Office", aliases=("Desk Lamp",))],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "desk lamp", "area": "Kitchen"},
        ),
        {"kind": "no_action", "response": "device_absent", "calls": [],
         "unavailable": {"area": "kitchen", "type": "desk lamp"}},
        id="alias-only-in-other-area-is-absent",
    ),
    pytest.param(
        _scenario(
            "Bedroom",
            [_entity("Bedroom Speaker", "media_players", "media_player", "Bedroom",
                     aliases=("Sleep Speaker",), device_class="speaker", features=("on", "off", "mute"))],
            "media_players", "mute", robustness="ambiguity", intent={"name": "sleep speaker"},
        ),
        {"kind": "action", "calls": [
            {"name": "HassMediaPlayerMute", "arguments": {
                "name": "Bedroom Speaker", "domain": ["media_player"], "device_class": ["speaker"]}},
        ]},
        id="alias-sleep-speaker-mute-resolves-canonical",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen",
                     aliases=("Overhead",), state="on")],
            "lights", "query_state", robustness="ambiguity", intent={"name": "overhead"},
        ),
        {"kind": "status",
         "calls": [{"name": "GetLiveContext",
                    "arguments": {"name": "Kitchen Ceiling Light", "domain": "light"}}],
         "state": "on"},
        id="alias-status-query-resolves-canonical",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Desk Lamp", "lights", "light", "Kitchen", features=("color_temp",))],
            "lights", "set_brightness", target=0,
        ),
        {"kind": "reject", "reason": "entity_lacks_capability", "calls": []},
        id="capability-light-without-brightness-rejects",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Desk Lamp", "lights", "light", "Kitchen", features=("color_temp",)),
                _entity("Kitchen Lamp", "lights", "light", "Kitchen", features=("brightness", "color_temp")),
            ],
            "lights", "set_brightness", target=0,
        ),
        {"kind": "reject", "reason": "entity_lacks_capability", "calls": []},
        id="capability-named-light-without-brightness-no-fallback",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [_entity("Living Room TV", "media_players", "media_player", "Living Room",
                     device_class="tv", features=("on", "off", "pause", "mute"))],
            "media_players", "volume_up", robustness="ambiguity", intent={"name": "tv"},
        ),
        {"kind": "reject", "reason": "entity_lacks_capability", "calls": []},
        id="capability-tv-without-volume-volume-up-rejects",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [
                _entity("Living Room TV", "media_players", "media_player", "Living Room",
                        device_class="tv", features=("on", "off", "pause", "volume")),
                _entity("Living Room Roku TV", "media_players", "media_player", "Living Room",
                        device_class="tv", features=("on", "off", "pause", "mute")),
            ],
            "media_players", "volume_up", robustness="ambiguity",
            intent={"name": "tv", "area": "Living Room"},
        ),
        {"kind": "no_action", "response": "clarify", "calls": [],
         "candidates": ["Living Room TV", "Living Room Roku TV"]},
        id="ambiguity-two-visible-tvs-one-capable-clarifies",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [
                _entity("Living Room TV", "media_players", "media_player", "Living Room",
                        device_class="tv", features=("on", "off", "volume", "mute")),
                _entity("Bedroom Speaker", "media_players", "media_player", "Bedroom",
                        device_class="speaker", features=("on", "off", "pause")),
            ],
            "media_players", "pause", target=0,
        ),
        {"kind": "reject", "reason": "entity_lacks_capability", "calls": []},
        id="capability-named-tv-without-pause-no-fallback",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [_entity("Living Room TV", "media_players", "media_player", "Living Room",
                     device_class="tv", features=("on", "off", "pause", "volume"))],
            "media_players", "pause", target=0,
        ),
        {"kind": "action", "calls": [
            {"name": "HassMediaPause", "arguments": {
                "name": "Living Room TV", "domain": ["media_player"], "device_class": ["tv"]}},
        ]},
        id="capability-tv-with-pause-pauses",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [_entity("Living Room TV", "media_players", "media_player", "Living Room",
                     device_class="tv", features=("on", "off", "pause", "volume"))],
            "media_players", "pause", target=0, robustness="unavailable",
        ),
        {"kind": "no_action", "response": "unsupported", "calls": [],
         "requested": {"kind": "action", "calls": [
             {"name": "HassMediaPause", "arguments": {
                 "name": "Living Room TV", "domain": ["media_player"], "device_class": ["tv"]}},
         ]},
         "unavailable_tools": ["HassMediaPause"]},
        id="capability-pause-tool-withheld-is-unsupported",
    ),
    pytest.param(
        _scenario(
            "Living Room",
            [_entity("Living Room Blinds", "covers", "cover", "Living Room",
                     device_class="blind", features=("open", "close"))],
            "covers", "set_position", target=0,
        ),
        {"kind": "no_action", "response": "unsupported", "calls": [],
         "blocker": "no position tool in Assist schema"},
        id="capability-covers-set-position-blocked",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Light North", "lights", "light", "Kitchen"),
                _entity("Kitchen Light South", "lights", "light", "Kitchen"),
            ],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "kitchen light"},
        ),
        {"kind": "no_action", "response": "clarify", "calls": [],
         "candidates": ["Kitchen Light North", "Kitchen Light South"]},
        id="ambiguity-two-matches-clarify-both",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Light One", "lights", "light", "Kitchen"),
                _entity("Kitchen Light Two", "lights", "light", "Kitchen"),
                _entity("Kitchen Light Three", "lights", "light", "Kitchen"),
                _entity("Kitchen Light Four", "lights", "light", "Kitchen"),
            ],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "kitchen light"},
        ),
        {"kind": "no_action", "response": "clarify", "calls": [],
         "candidates": ["Kitchen Light One", "Kitchen Light Two", "Kitchen Light Three"]},
        id="ambiguity-four-matches-candidates-capped-at-three",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Light North", "lights", "light", "Kitchen"),
                _entity("Desk Lamp", "lights", "light", "Kitchen"),
            ],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "north"},
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOn", "arguments": {"name": "Kitchen Light North", "domain": ["light"]}},
        ]},
        id="ambiguity-one-match-acts-by-name",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Ceiling Light", "lights", "light", "Kitchen"),
                _entity("Kitchen Island Light", "lights", "light", "Kitchen"),
            ],
            "lights", "turn_off", targeting="area", area="Kitchen",
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"area": "Kitchen", "domain": ["light"]}},
        ]},
        id="area-multiple-lights-use-area-argument",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen")],
            "lights", "turn_off", targeting="area", area="Kitchen",
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"name": "Kitchen Ceiling Light", "domain": ["light"]}},
        ]},
        id="area-single-light-uses-name",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Speaker", "media_players", "media_player", "Kitchen",
                     device_class="speaker", features=("on", "off", "volume"))],
            "media_players", "pause", targeting="area", area="Kitchen",
        ),
        {"kind": "reject", "reason": "entity_lacks_capability", "calls": []},
        id="area-no-capable-media-player-rejects",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen")],
            "lights", "turn_on", targeting="area", area="Garage",
        ),
        {"kind": "no_action", "response": "area_unavailable", "calls": [],
         "unavailable": {"area": "garage", "type": "lights"}},
        id="area-without-lights-is-unavailable",
    ),
    pytest.param(
        _scenario(
            "Hallway",
            [
                _entity("Kitchen Light North", "lights", "light", "Kitchen"),
                _entity("Kitchen Light South", "lights", "light", "Kitchen"),
                _entity("Hallway Light", "lights", "light", "Hallway"),
            ],
            "lights", "turn_on", robustness="ambiguity", intent={"name": "light", "area": "Kitchen"},
        ),
        {"kind": "no_action", "response": "clarify", "calls": [],
         "candidates": ["Kitchen Light North", "Kitchen Light South"]},
        id="ambiguity-intent-area-narrows-candidates",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen")],
            "locks", "unlock", robustness="ambiguity", intent={"name": "front door"},
        ),
        {"kind": "no_action", "response": "device_absent", "calls": [],
         "unavailable": {"area": "kitchen", "type": "front door"}},
        id="ambiguity-absent-device-in-area",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen")],
            "lights", "turn_on", target=0,
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOn", "arguments": {"name": "Kitchen Ceiling Light", "domain": ["light"]}},
        ]},
        id="ordinary-light-turn-on",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [_entity("Coffee Maker Plug", "switches", "switch", "Kitchen", device_class="outlet", state="on")],
            "switches", "turn_off", target=0,
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"name": "Coffee Maker Plug", "domain": ["switch"]}},
        ]},
        id="ordinary-switch-turn-off",
    ),
    pytest.param(
        _scenario(
            "Office",
            [_entity("Office Fan", "fans", "fan", "Office", features=("on", "off", "percentage"), state="off")],
            "fans", "query_state", target=0,
        ),
        {"kind": "status",
         "calls": [{"name": "GetLiveContext", "arguments": {"name": "Office Fan", "domain": "fan"}}],
         "state": "off"},
        id="ordinary-fan-status-query",
    ),
    pytest.param(
        _scenario(
            "Kitchen",
            [
                _entity("Kitchen Ceiling Light", "lights", "light", "Kitchen"),
                _entity("Desk Lamp", "lights", "light", "Kitchen"),
            ],
            "lights", "turn_off", targeting="multiple", targets=[0, 1],
        ),
        {"kind": "action", "calls": [
            {"name": "HassTurnOff", "arguments": {"name": "Kitchen Ceiling Light", "domain": ["light"]}},
            {"name": "HassTurnOff", "arguments": {"name": "Desk Lamp", "domain": ["light"]}},
        ], "script_targets": []},
        id="ordinary-two-lights-turn-off-multiple",
    ),
]


@pytest.mark.parametrize(("scenario", "expected"), CASES)
def test_gold_label_matches_hand_label(scenario, expected):
    assert gold_from_scenario(scenario, random.Random(0)) == expected


def test_person_entity_outside_exposed_domains_is_refused():
    home = {
        "sayso_entity_area": "Kitchen",
        "entities": [_entity("Kitchen Ceiling Light", "lights", "light", "Kitchen")],
        "context_entities": [{"name": "Matt", "domain": "person", "area": "Kitchen"}],
    }
    assert expected_unexposed_domain(home, random.Random(0)) == {
        "kind": "no_action",
        "response": "refuse",
        "calls": [],
        "subfamily": "unexposed_domain",
        "unavailable": {"domain": "person", "name": "Matt"},
    }
