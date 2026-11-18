from __future__ import annotations

import random
from functools import lru_cache

from generators.homes import _eval_cases_module
from generators.shared_alias import HOME_SIZES, build_generic_phrase_spec
from generators.validation import check_quality_eval_overlap, shared_alias_action_conflict, validate_spec


def _entity(name: str, area: str = "Bedroom", capability: str = "lights") -> dict:
    return {"name": name, "area": area, "aliases": [name], "capability": capability}


def _bedside_home() -> list[dict]:
    return [_entity("Robin's Bedside Lamp"), _entity("Bedroom Bedside Lamp")]


def _action(utterance: str, calls: list[dict], entities: list[dict] | None = None) -> dict:
    return {
        "utterance": utterance,
        "home": {"entities": entities if entities is not None else _bedside_home()},
        "expected": {"kind": "action", "calls": calls},
    }


def _named(name: str) -> dict:
    return {"name": "HassTurnOff", "arguments": {"name": name}}


def _area_wide(area: str) -> dict:
    return {"name": "HassTurnOff", "arguments": {"area": area, "domain": ["light"]}}


def _eval_ambiguity_utterances() -> tuple[str, ...]:
    cases = _eval_cases_module()
    return tuple(case.utterance.casefold() for case in cases.load_all_cases() if case.category == "ambiguity")


@lru_cache(maxsize=1)
def _generic_specs() -> tuple[dict, ...]:
    return tuple(
        spec for index in range(400)
        if (spec := build_generic_phrase_spec(random.Random(index), index,
                                              HOME_SIZES[index % len(HOME_SIZES)])) is not None
    )


def test_generic_phrase_naming_two_same_area_entities_is_flagged():
    spec = _action("switch the bedside lamp off in the bedroom", [_named("Robin's Bedside Lamp")])
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_full_name_or_sole_possessive_is_not_flagged():
    full_name = _action("switch robin's bedside lamp off", [_named("Robin's Bedside Lamp")])
    possessive = _action("switch off the bedside lamp that's Robin's", [_named("Robin's Bedside Lamp")])
    for spec in (full_name, possessive):
        assert not shared_alias_action_conflict(spec)
        assert validate_spec(spec) != "shared_alias_action"


def test_area_wide_call_covering_both_sharers_is_not_flagged():
    spec = _action("switch the bedside lamp off in the bedroom", [_area_wide("Bedroom")])
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) != "shared_alias_action"


def test_area_wide_call_on_another_area_does_not_cover_the_sharers():
    spec = _action("switch the bedside lamp off in the bedroom", [_area_wide("Kitchen")])
    assert shared_alias_action_conflict(spec)


def test_single_word_phrase_is_not_flagged():
    entities = [_entity("Robin's Lamp"), _entity("Bedroom Lamp"), *_bedside_home()]
    spec = _action("switch the lamp off in the bedroom", [_named("Robin's Lamp")], entities)
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) != "shared_alias_action"


def test_built_generic_phrase_specs_name_only_real_same_area_candidates():
    specs = _generic_specs()
    assert specs
    assert {spec["capability"] for spec in specs} == {"lights", "fans"}
    for spec in specs:
        tail = spec["shared_alias"]
        entities = spec["home"]["entities"]
        by_name = {entity["name"]: entity for entity in entities}
        candidates = spec["expected"]["candidates"]
        assert spec["subcategory"] == "generic_phrase"
        assert 2 <= len(candidates) <= 3
        areas = {by_name[name]["area"] for name in candidates}
        assert len(areas) == 1
        area = areas.pop()
        expected_sharers = {
            entity["name"] for entity in entities
            if entity["area"] == area and entity["name"].casefold().endswith(f" {tail}")
        }
        assert set(candidates) == expected_sharers
        assert all(by_name[name]["capability"] == spec["capability"] for name in candidates)
        assert spec["capability"] in {"lights", "fans"}
        assert spec["expected"]["kind"] == "no_action"
        assert spec["expected"]["calls"] == []
        utterance = spec["utterance"].casefold()
        assert tail in utterance
        assert "'" not in utterance
        assert not any(name.casefold() in utterance for name in by_name)
        assert not shared_alias_action_conflict(spec)
        assert validate_spec(spec) is None
        assert not check_quality_eval_overlap(spec["utterance"])


def test_generated_generic_phrases_avoid_eval_ambiguity_phrases():
    ambiguous = _eval_ambiguity_utterances()
    assert ambiguous
    for spec in _generic_specs():
        tail = spec["shared_alias"]
        assert tail not in {"bedside lamp", "reading lamp"}
        assert not any(tail in text for text in ambiguous)
        assert spec["utterance"].casefold() not in ambiguous


def test_generic_phrase_specs_are_deterministic_per_seed():
    first = [(spec["utterance"], spec["expected"]["candidates"]) for spec in _generic_specs()]
    second = [
        (spec["utterance"], spec["expected"]["candidates"])
        for index in range(400)
        if (spec := build_generic_phrase_spec(random.Random(index), index,
                                              HOME_SIZES[index % len(HOME_SIZES)])) is not None
    ]
    assert first == second
