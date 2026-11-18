from __future__ import annotations

import random
import re

from generators.context import exposed_entities
from generators.homes import SUFFIX_LISTED_RATE, apply_suffix_aliases, make_entity, suffix_alias
from generators.ohf import _grammar, render_call
from generators.scenarios.core import _usable_aliases, build_scenario
from generators.validation import validate_spec

PURIFIER = "Levoit VeSync Core 300S Plus Smart Air Purifier"
SEEDS = range(40)
DETERMINER_BAD = re.compile(r"\b(the the|my the|in my the|in our the|our the|my my|our our)\b", re.I)


def _entity(name: str, area: str = "Office") -> dict:
    return make_entity(name=name, capability="fans", area=area, floor="Main Floor", rng=random.Random(name))


def _fan_call(**arguments) -> dict:
    return {"name": "HassFanSetSpeed", "arguments": {"percentage": 50, **arguments}}


def _rng(listed: bool) -> random.Random:
    seed = 0
    while (random.Random(seed).random() < SUFFIX_LISTED_RATE) != listed:
        seed += 1
    return random.Random(seed)


def _context_names(home: dict, name: str) -> list[str]:
    for info in exposed_entities(home):
        if info["names"] and name in info["names"].split(", "):
            return [part.strip() for part in info["names"].split(", ")]
    return []


def test_fan_speed_renders_for_name_area_and_floor_forms() -> None:
    for seed in SEEDS:
        name_text = render_call(_fan_call(name=PURIFIER), PURIFIER, str(seed))
        area_text = render_call(_fan_call(area="Kitchen"), "", str(seed))
        floor_text = render_call(_fan_call(floor="Main Floor"), "", str(seed))
        assert name_text and PURIFIER in name_text and "50" in name_text, name_text
        assert area_text and "Kitchen" in area_text and "50" in area_text, area_text
        assert floor_text and "Main Floor" in floor_text and "50" in floor_text, floor_text


def test_suffix_spoken_target_renders_and_gold_keeps_full_name() -> None:
    for seed in SEEDS:
        text = render_call(_fan_call(name=PURIFIER), "air purifier", str(seed))
        assert text and "air purifier" in text.casefold() and PURIFIER not in text, text


def test_listed_home_yields_air_purifier_alias() -> None:
    entities = [_entity(PURIFIER)]
    apply_suffix_aliases(entities, _rng(listed=True))
    assert suffix_alias(PURIFIER) == "air purifier"
    assert "air purifier" in entities[0]["aliases"]
    assert entities[0]["name"] == PURIFIER


def test_unlisted_home_keeps_suffix_out_of_aliases_and_context() -> None:
    entities = [_entity(PURIFIER)]
    apply_suffix_aliases(entities, _rng(listed=False))
    assert "air purifier" not in entities[0].get("aliases", [])
    names = _context_names({"entities": entities}, PURIFIER)
    assert names and "air purifier" not in [name.casefold() for name in names]


def test_two_purifiers_share_no_suffix_alias() -> None:
    for listed in (True, False):
        entities = [_entity(PURIFIER), _entity("Winix Smart Air Purifier", area="Bedroom")]
        apply_suffix_aliases(entities, _rng(listed=listed))
        for entity in entities:
            assert "air purifier" not in [alias.casefold() for alias in entity.get("aliases", [])]
            assert entity["name"] in (PURIFIER, "Winix Smart Air Purifier")


def test_late_added_entity_blocks_suffix_alias_at_scenario_time() -> None:
    entities = [_entity(PURIFIER)]
    apply_suffix_aliases(entities, _rng(listed=True))
    home = {"entities": [*entities, _entity("Winix Smart Air Purifier", area="Bedroom")]}
    assert "air purifier" not in _usable_aliases(entities[0], home)


def test_short_names_get_no_suffix_alias() -> None:
    assert suffix_alias("Ceiling Fan") is None
    assert suffix_alias("Office Desk Lamp") is None


def test_suffix_rows_sampled_with_unlisted_majority_and_full_name_gold() -> None:
    suffix_rows = 0
    unlisted_rows = 0
    for index in range(1200):
        scenario = build_scenario(
            index=index, seed=7, capability="fans", operation="set_speed", home_size=24, robustness="ordinary",
        )
        target = scenario.get("target_entity")
        if not target or not suffix_alias(target["name"]):
            continue
        spoken = (scenario.get("spoken_targets") or {}).get(target["name"])
        if not spoken:
            continue
        assert spoken == suffix_alias(target["name"])
        suffix_rows += 1
        names = _context_names(scenario["home"], target["name"])
        listed = spoken in target.get("aliases", [])
        assert (spoken in [name.casefold() for name in names]) == listed, names
        if not listed:
            unlisted_rows += 1
        calls = scenario["expected"]["calls"]
        assert any((call.get("arguments") or {}).get("name") == target["name"] for call in calls)
        assert validate_spec(scenario) is None
    assert suffix_rows >= 20
    assert unlisted_rows / suffix_rows >= 0.5


def test_no_doubled_determiner_in_area_or_floor_renders() -> None:
    _, entries, _ = _grammar()
    literals = ("the Kitchen", "my Kitchen", "our Kitchen", "Kitchen", "my Main Floor", "Main Floor")
    checked = 0
    for block, _sentence, _expression, refs in entries:
        slots = {ref.slot_name for ref in refs}
        if slots not in ({"area"}, {"floor"}):
            continue
        key = slots.pop()
        for literal in literals:
            for seed in range(30):
                text = render_call({"name": block["intent"], "arguments": {key: literal}}, "", str(seed))
                if text is None:
                    continue
                checked += 1
                assert not DETERMINER_BAD.search(text), text
    assert checked > 0
