from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from adapters.schema import tool_schema_map, validate_tool_arguments
from generators.gold import gold_from_scenario
from generators.row_generation import _DATETIME_UTTERANCES, _datetime_utterance
from generators.tools import build_call_for_operation
from generators.utterances import (
    _AREA_ANY_STATUS_PHRASES,
    _AREA_DOMAIN_STATUS_PHRASES,
    _AREA_STATUS_WORDS,
    _DATETIME_PHRASES,
    _ENTITY_STATUS_PHRASES,
    area_status_phrasing,
    datetime_phrasing,
    entity_status_phrasing,
    media_search_phrasing,
    request_seed_from_spec,
)
from generators.validation import _STATUS_QUERY

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QUERY_CAPABILITIES = ("lights", "switches", "fans", "covers", "locks", "media_players", "climate")
AREAS = ("Living Room", "Kitchen", "Bedroom", "Office")
SEEDS = range(500)
LIVE_PROBES = (
    "what time is it",
    "what's the date today",
    "what day is it",
    "is anything on in the living room",
    "is anything on in the bedroom",
    "are the lights on in the kitchen",
    "which lights are on in the kitchen",
    "is the tv on",
    "is the T V on",
    "play some jazz on the kitchen echo dot",
    "play the tv",
    "turn off the kitchen light",
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", "", text.lower()).split())


def _schemas() -> dict:
    schema = json.loads((FIXTURES / "sayso_tool_schema_v2.json").read_text(encoding="utf-8"))
    return tool_schema_map(schema["tools"])


def _area_query_calls():
    for seed in SEEDS:
        capability = QUERY_CAPABILITIES[seed % len(QUERY_CAPABILITIES)]
        area = AREAS[seed % len(AREAS)]
        yield area, capability, build_call_for_operation(
            None, capability, "query_state", random.Random(seed), area=area
        )


def test_area_query_calls_carry_area_and_optional_domain_only() -> None:
    forms = set()
    for area, capability, call in _area_query_calls():
        assert call["name"] == "GetLiveContext"
        arguments = call["arguments"]
        assert arguments["area"] == area
        assert "name" not in arguments
        assert set(arguments) <= {"area", "domain"}
        forms.add("domain" in arguments)
    assert forms == {True, False}


def test_area_query_calls_validate_against_pinned_schema() -> None:
    schemas = _schemas()
    for _, _, call in _area_query_calls():
        assert validate_tool_arguments(call["name"], call["arguments"], schemas) is None, call


def test_no_query_call_puts_an_area_name_in_name() -> None:
    for area, _, call in _area_query_calls():
        assert call["arguments"].get("name") not in AREAS
        assert area not in str(call["arguments"].get("name", ""))


def test_area_status_utterances_mention_the_area() -> None:
    for seed, (area, capability, call) in enumerate(_area_query_calls()):
        spec = {"candidate_id": f"area-status-{seed}", "expected": {"kind": "status", "calls": [call]}}
        utterance = request_seed_from_spec(spec)
        assert area.casefold() in utterance, (capability, call, utterance)


def test_each_intent_has_at_least_15_distinct_phrasings() -> None:
    datetime = {datetime_phrasing(random.Random(seed)) for seed in SEEDS}
    entity = {entity_status_phrasing("the kitchen light", random.Random(seed)) for seed in SEEDS}
    area = {
        area_status_phrasing({"area": "Kitchen", "domain": "light"}, random.Random(seed))
        for seed in SEEDS
    }
    area_any = {area_status_phrasing({"area": "Kitchen"}, random.Random(seed)) for seed in SEEDS}
    media = {
        media_search_phrasing("the kitchen echo dot", "jazz", random.Random(seed))
        for seed in SEEDS
    }
    assert len(datetime) >= 15, sorted(datetime)
    assert len(entity) >= 15, sorted(entity)
    assert len(area) >= 15, sorted(area)
    assert len(area_any) >= 15, sorted(area_any)
    assert len(media) >= 15, sorted(media)


def test_no_live_probe_utterance_appears_in_generated_phrasings() -> None:
    probes = {_normalize(probe) for probe in LIVE_PROBES}
    generated: list[str] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        generated.append(datetime_phrasing(rng))
        generated.append(entity_status_phrasing("the living room tv", rng))
        generated.append(entity_status_phrasing("the kitchen light", rng))
        for area in AREAS:
            for domain in (None, "light", "switch", "fan", "media_player"):
                arguments = {"area": area} if domain is None else {"area": area, "domain": domain}
                generated.append(area_status_phrasing(arguments, rng))
        generated.append(media_search_phrasing("the kitchen echo dot", "jazz", rng))
        generated.append(media_search_phrasing("the living room tv", "the news", rng))
    leaked = sorted({text for text in generated if _normalize(text) in probes})
    assert leaked == []


def _datetime_category(text: str) -> str | None:
    text = text.lower()
    if re.search(r"\bdate\b", text):
        return "date"
    if re.search(r"\b(day|weekday|tuesday)\b", text):
        return "day"
    if re.search(r"\btime\b", text):
        return "time"
    return None


def test_datetime_pool_contains_the_shared_datetime_phrasings() -> None:
    assert set(_DATETIME_PHRASES) <= set(_DATETIME_UTTERANCES)


def test_datetime_rows_cover_time_date_and_day_evenly() -> None:
    assert None not in {_datetime_category(text) for text in _DATETIME_UTTERANCES}
    counts = Counter(
        _datetime_category(_datetime_utterance(random.Random(seed), seed)) for seed in range(600)
    )
    for category in ("time", "date", "day"):
        assert counts[category] >= 0.2 * 600, counts


def test_datetime_pool_avoids_live_probes_except_existing_time_probe() -> None:
    probes = {_normalize(probe) for probe in LIVE_PROBES if probe != "what time is it"}
    assert not {_normalize(text) for text in _DATETIME_UTTERANCES} & probes


def test_datetime_gold_arguments_are_empty() -> None:
    scenario = {"family": "datetime", "capability": "datetime"}
    for seed in SEEDS:
        expected = gold_from_scenario(scenario, random.Random(seed))
        assert expected == {"kind": "action", "calls": [{"name": "GetDateTime", "arguments": {}}]}


def test_domain_scoped_area_status_never_uses_area_only_phrases() -> None:
    area_only = {phrase.format(area="kitchen") for phrase in _AREA_ANY_STATUS_PHRASES}
    for domain, (state, noun) in _AREA_STATUS_WORDS.items():
        allowed = {phrase.format(area="kitchen", state=state, noun=noun) for phrase in _AREA_DOMAIN_STATUS_PHRASES}
        outputs = {
            area_status_phrasing({"area": "Kitchen", "domain": [domain]}, random.Random(seed))
            for seed in SEEDS
        }
        assert outputs <= allowed, (domain, sorted(outputs - allowed))
        assert not outputs & area_only, (domain, sorted(outputs & area_only))


def test_area_only_status_still_uses_area_phrases() -> None:
    area_only = {phrase.format(area="kitchen") for phrase in _AREA_ANY_STATUS_PHRASES}
    outputs = {area_status_phrasing({"area": "Kitchen"}, random.Random(seed)) for seed in SEEDS}
    assert outputs <= area_only


def test_every_status_phrasing_matches_the_status_query_validator() -> None:
    formatted = [phrase.format("the kitchen light") for phrase in _ENTITY_STATUS_PHRASES]
    formatted += [
        phrase.format(area="kitchen", state=state, noun=noun)
        for state, noun in _AREA_STATUS_WORDS.values()
        for phrase in _AREA_DOMAIN_STATUS_PHRASES
    ]
    formatted += [phrase.format(area="kitchen") for phrase in _AREA_ANY_STATUS_PHRASES]
    misses = [text for text in formatted if not _STATUS_QUERY.search(text.lower())]
    assert misses == []


def test_entity_status_phrasings_name_status_or_state() -> None:
    for phrase in _ENTITY_STATUS_PHRASES:
        lowered = phrase.lower()
        assert "status" in lowered or "state" in lowered, phrase


def test_status_pools_keep_at_least_15_distinct_phrasings() -> None:
    assert len(_ENTITY_STATUS_PHRASES) >= 15
    assert len(set(_ENTITY_STATUS_PHRASES)) == len(_ENTITY_STATUS_PHRASES)
    assert len(set(_AREA_DOMAIN_STATUS_PHRASES)) >= 15
    assert len(set(_AREA_ANY_STATUS_PHRASES)) >= 15
