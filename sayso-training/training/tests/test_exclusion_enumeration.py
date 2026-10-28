from __future__ import annotations

import pytest

from generators.scenarios import area as area_scenarios
from generators.utterances import _leave_clause
from generators.validation import check_fuzzy_eval_overlap

_INDICES = range(40)


def _spec(index: int) -> dict:
    return area_scenarios.build_area_spec("exclusion_within_area", index, seed=20260917)


@pytest.mark.parametrize("index", _INDICES)
def test_area_exclusion_calls_only_real_home_entities(index: int) -> None:
    spec = _spec(index)
    home_names = {entity["name"] for entity in spec["home"]["entities"]}
    kept = spec["excluded_names"][0]
    calls = spec["expected"]["calls"]
    assert len(calls) >= 2
    for call in calls:
        arguments = call["arguments"]
        assert arguments["name"] in home_names
        assert arguments["name"] != kept
        assert "area" not in arguments


@pytest.mark.parametrize("index", _INDICES)
def test_area_exclusion_enumerates_every_other_light_in_the_area(index: int) -> None:
    spec = _spec(index)
    satellite = spec["home"]["sayso_entity_area"]
    kept = spec["excluded_names"][0]
    in_area = sorted(
        entity["name"]
        for entity in spec["home"]["entities"]
        if entity["capability"] == "lights" and entity["area"] == satellite and entity["name"] != kept
    )
    called = sorted(call["arguments"]["name"] for call in spec["expected"]["calls"])
    assert called == in_area


@pytest.mark.parametrize("index", _INDICES)
def test_area_exclusion_utterance_keeps_leave_and_the_kept_name(index: int) -> None:
    spec = _spec(index)
    utterance = spec["utterance"].casefold()
    assert "leave" in utterance
    assert spec["excluded_names"][0].casefold() in utterance


def test_named_exclusion_phrasing_variants_appear_and_keep_leave() -> None:
    excluded = ["Kitchen Accent Light"]
    clauses = {_leave_clause(excluded, {"candidate_id": f"row_{index}"}) for index in range(200)}
    assert len(clauses) >= 4
    for clause in clauses:
        assert "leave" in clause
        assert "Kitchen Accent Light" in clause


def test_named_exclusion_phrasing_variants_pass_the_eval_overlap_check() -> None:
    for clause in ("unchanged", "as it is", "how it is"):
        assert not check_fuzzy_eval_overlap(f"turn off the kitchen lights, but leave Kitchen Accent Light {clause}")
    assert not check_fuzzy_eval_overlap("turn off the kitchen lights, but not Kitchen Accent Light, leave it")
