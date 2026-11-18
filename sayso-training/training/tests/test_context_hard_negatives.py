import copy
import json
import random
from dataclasses import asdict
from unittest.mock import patch

import pytest

from generators import row_generation
from generators.config import GeneratorConfig, default_allocations
from generators.context import exposed_entities, serialize_context
from generators.deduplication import DuplicateTracker
from generators.gold import expected_no_action, gold_matches_family
from generators.planning import build_plan
from generators.validation import validate_spec


MAX_ATTEMPTS = 200


def _attempt(config, tracker, index, attempt):
    slot = {
        "index": index,
        "family": "unsupported",
        "capability": "media_players",
        "operation": "turn_on",
        "home_size": 16,
    }
    return row_generation.generate_row(
        slot, config, random.Random(config.seed + index * MAX_ATTEMPTS + attempt),
        excluded=set(), dup_tracker=tracker, attempt=attempt,
        stt_remaining=config.count, stt_log_remaining=config.count, rows_remaining=config.count,
    )


def _sample(count=300):
    config = GeneratorConfig(count=count, seed=812, stt_noise_rate=0)
    tracker = DuplicateTracker()
    results = []
    attempts = 0
    with patch.object(row_generation, "render_example", wraps=row_generation.render_example) as render:
        for index in range(count):
            for attempt in range(MAX_ATTEMPTS):
                attempts += 1
                row, reason = _attempt(config, tracker, index, attempt)
                if row is not None:
                    results.append((row, copy.deepcopy(render.call_args.args[0])))
                    break
                assert reason == "entity_lacks_capability", reason
            else:
                raise AssertionError(f"slot {index} produced no row in {MAX_ATTEMPTS} attempts")
    return results, attempts


def _first_attempts(count):
    config = GeneratorConfig(count=count, seed=812, stt_noise_rate=0)
    tracker = DuplicateTracker()
    return [_attempt(config, tracker, index, 0) for index in range(count)]


@pytest.fixture(scope="module")
def sample():
    return _sample()


def _is_context_refusal(spec):
    names = {entity["name"] for entity in spec["home"].get("context_entities", [])}
    return bool(names.intersection(spec["expected"].get("unsupported_names", [])))


def test_context_hard_negative_share_and_labels(sample):
    results, attempts = sample
    assert len(results) == 300
    negatives = [(row, spec) for row, spec in results if _is_context_refusal(spec)]
    assert len(negatives) == len(results)
    assert 0.25 <= row_generation.CONTEXT_HARD_NEGATIVE_RATE <= 0.35
    assert len(negatives) / attempts == pytest.approx(
        row_generation.CONTEXT_HARD_NEGATIVE_RATE, abs=0.07
    )
    for row, spec in negatives:
        expected = spec["expected"]
        name, = expected["unsupported_names"]
        assert expected == expected_no_action("device_unsupported", unsupported_names=[name])
        assert gold_matches_family(expected, "unsupported")
        assert validate_spec(spec) is None
        assert name not in {entity["name"] for entity in spec["home"]["entities"]}
        assert name in {entity["names"] for entity in exposed_entities(spec["home"])}
        assert name in spec["utterance"]
        assert name in row["messages"][1]["content"]
        assert serialize_context(spec["home"], spec["utterance"]) == row["messages"][0]["content"]
        assert name in row["messages"][0]["content"]
        assert row["messages"][-1]["content"] == f"{name} does not support that."
        assert not any(message.get("tool_calls") for message in row["messages"])
        assert not row["metadata"]["unavailable_tools"]
        assert row["metadata"]["capability"] == "media_players"
        assert row["metadata"]["operation"] == "turn_on"
        assert spec["linguistics"] == [{"source": "sayso_fallback", "intent": "device_unsupported"}]
        assert "request_hint" not in spec
        assert "clarify_question" not in spec
        assert not spec.get("stt_corruption")


def test_context_hard_negatives_are_deterministic(sample):
    assert _sample() == sample


@pytest.mark.parametrize("domain", ["person", "sensor", "binary_sensor", "sun", "select"])
def test_context_name_is_valid_for_refusal_but_not_action(domain):
    name = "Context Entry"
    spec = {
        "capability": "media_players",
        "operation": "turn_on",
        "home": {"entities": [], "context_entities": [{"name": name, "domain": domain}]},
        "expected": expected_no_action("device_unsupported", unsupported_names=[name]),
    }
    assert validate_spec(spec) is None
    calls = [{"name": "HassTurnOn", "arguments": {"name": name}}]
    spec["expected"]["calls"] = calls
    assert validate_spec(spec) == "no_action_has_calls"
    spec["expected"] = {"kind": "action", "calls": calls}
    assert validate_spec(spec) == "unknown_canonical_entity"
    spec["expected"] = expected_no_action("device_unsupported", unsupported_names=["Unknown Entry"])
    assert validate_spec(spec) == "unknown_canonical_entity"


def test_home_without_context_entities_is_unchanged(monkeypatch):
    original = row_generation.scenario_to_spec

    def without_context(scenario):
        spec = original(scenario)
        spec["home"].pop("context_entities", None)
        return spec

    monkeypatch.setattr(row_generation, "scenario_to_spec", without_context)
    monkeypatch.setattr(row_generation, "CONTEXT_HARD_NEGATIVE_RATE", 0.0)
    before = _first_attempts(8)
    monkeypatch.setattr(row_generation, "CONTEXT_HARD_NEGATIVE_RATE", 1.0)
    assert _first_attempts(8) == before
    assert all(row is None for row, _ in before)
    assert {reason for _, reason in before} == {"entity_lacks_capability"}


def test_other_families_and_rng_are_unchanged(monkeypatch):
    config = GeneratorConfig(count=140, seed=812, stt_noise_rate=0)
    plan = build_plan(140, config.seed, default_allocations(), datetime_required=1)
    for slot in plan.slots:
        if slot.family == "unsupported":
            continue
        results = []
        for rate in (0.0, 1.0):
            monkeypatch.setattr(row_generation, "CONTEXT_HARD_NEGATIVE_RATE", rate)
            rng = random.Random(config.seed + slot.index)
            row, reason = row_generation.generate_row(
                asdict(slot), config, rng, excluded=set(), dup_tracker=DuplicateTracker(),
            )
            results.append(json.dumps((row, reason, rng.getstate()), sort_keys=True))
        assert results[0] == results[1], slot.family
