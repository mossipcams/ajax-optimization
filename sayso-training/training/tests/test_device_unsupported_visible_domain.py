import random

from generators.gold import _ambiguous_gold, gold_from_scenario
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.row_generation import generate_row
from generators.validation import validate_spec


def _entity(name="Kitchen Lamp", domain="light", capability="lights", area="Kitchen", features=()):
    return {
        "entity_id": f"{domain}.{name.casefold().replace(' ', '_')}",
        "name": name,
        "aliases": [name],
        "domain": domain,
        "kind": domain,
        "capability": capability,
        "device_class": None,
        "area": area,
        "floor": "Main Floor",
        "state": "off",
        "capabilities": list(features),
    }


def _home(entity):
    return {"sayso_entity_area": entity["area"], "entities": [entity]}


def _scenario(home, capability="lights", operation="set_color", family="ordinary"):
    return {
        "family": family,
        "home": home,
        "capability": capability,
        "operation": operation,
        "targeting": "area",
    }


def test_area_path_same_domain_without_feature_yields_action():
    home = _home(_entity(features=("on", "off")))
    expected = gold_from_scenario(_scenario(home, family="unsupported"), random.Random(1))
    assert expected["kind"] == "action"
    call = expected["calls"][0]
    assert call["name"] == "HassLightSet"
    assert call["arguments"]["name"] == "Kitchen Lamp"


def test_area_path_same_domain_without_feature_non_unsupported_family_device_unsupported():
    home = _home(_entity())
    expected = gold_from_scenario(_scenario(home, family="ordinary"), random.Random(1))
    assert expected["kind"] == "no_action"
    assert expected["response"] == "device_unsupported"


def test_area_path_cross_domain_still_device_unsupported():
    home = _home(_entity(name="Kitchen Temperature", domain="sensor"))
    expected = gold_from_scenario(_scenario(home), random.Random(1))
    assert expected["kind"] == "no_action"
    assert expected["response"] == "device_unsupported"


def test_ambiguous_same_domain_without_feature_yields_action():
    home = _home(_entity(features=("on", "off")))
    expected = _ambiguous_gold(home, "lights", "set_color", random.Random(1), family="unsupported")
    assert expected["kind"] == "action"
    call = expected["calls"][0]
    assert call["name"] == "HassLightSet"
    assert call["arguments"]["name"] == "Kitchen Lamp"


def test_ambiguous_same_domain_without_feature_non_unsupported_family_device_unsupported():
    home = _home(_entity())
    expected = _ambiguous_gold(home, "lights", "set_color", random.Random(1), family="ordinary")
    assert expected["kind"] == "no_action"
    assert expected["response"] == "device_unsupported"


def test_ambiguous_cross_domain_still_device_unsupported():
    home = _home(_entity(name="Kitchen Temperature", domain="sensor"))
    expected = _ambiguous_gold(home, "lights", "set_color", random.Random(1))
    assert expected["kind"] == "no_action"
    assert expected["response"] == "device_unsupported"


def test_unsupported_family_same_domain_slot_yields_action_row():
    config = GeneratorConfig(count=1, seed=20261024, stt_noise_rate=0)
    tracker = DuplicateTracker()
    slot = {
        "index": 3,
        "family": "unsupported",
        "capability": "lights",
        "operation": "set_color",
        "home_size": 8,
        "robustness": "ambiguity",
    }
    rng = random.Random(config.seed)
    row, reason = None, None
    for attempt in range(20):
        row, reason = generate_row(
            slot, config, rng, excluded=set(), dup_tracker=tracker, attempt=attempt,
        )
        if row is not None:
            break
    assert row is not None, f"slot dropped: {reason}"
    calls = [
        call
        for message in row["messages"]
        for call in message.get("tool_calls") or []
    ]
    assert calls
    (call,) = calls
    assert call["function"]["name"] == "light__HassLightSet"


def _light_action_spec(entity):
    return {
        "capability": "lights",
        "operation": "set_color",
        "utterance": "Set the Kitchen Lamp to red",
        "home": {"entities": [entity], "sayso_entity_area": entity["area"]},
        "expected": {
            "kind": "action",
            "calls": [{"name": "HassLightSet", "arguments": {"name": entity["name"]}}],
        },
    }


def test_validation_accepts_same_domain_call_without_feature():
    spec = {**_light_action_spec(_entity(features=("on", "off"))), "family": "unsupported"}
    assert validate_spec(spec) is None


def test_validation_rejects_same_domain_call_without_feature_or_family():
    assert validate_spec(_light_action_spec(_entity())) == "entity_lacks_capability"


def test_validation_rejects_cross_domain_call():
    spec = _light_action_spec(_entity(name="Kitchen Temperature", domain="sensor"))
    assert validate_spec(spec) == "entity_lacks_capability"
