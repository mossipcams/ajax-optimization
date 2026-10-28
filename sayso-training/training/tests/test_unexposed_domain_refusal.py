import copy
import random
from collections import defaultdict
from unittest.mock import patch

import pytest

from evals.scorer import is_refusal
from generators import row_generation
from generators.config import GeneratorConfig
from generators.context import exposed_entities, serialize_context
from generators.deduplication import DuplicateTracker
from generators.gold import expected_no_action, expected_unexposed_domain, gold_matches_family
from generators.validation import check_quality_eval_overlap, validate_spec

DOMAINS = {"person", "sun"}


def _sample(count=512):
    config = GeneratorConfig(count=count, seed=20261023, stt_noise_rate=0)
    tracker = DuplicateTracker()
    rng = random.Random(config.seed)
    results = []
    with patch.object(row_generation, "render_example", wraps=row_generation.render_example) as render:
        for index in range(count):
            slot = {
                "index": index,
                "family": "absence",
                "capability": "lights",
                "operation": "turn_on",
                "home_size": 8,
            }
            row = None
            for attempt in range(20):
                row, _ = row_generation.generate_row(
                    slot, config, rng, excluded=set(), dup_tracker=tracker, attempt=attempt,
                )
                if row is not None:
                    break
            assert row is not None
            results.append((row, copy.deepcopy(render.call_args.args[0])))
    return results


@pytest.fixture(scope="module")
def sample():
    return _sample()


def _is_unexposed(spec):
    return spec["expected"].get("subfamily") == "unexposed_domain"


def test_unexposed_domain_share_and_labels(sample):
    refusals = [(row, spec) for row, spec in sample if _is_unexposed(spec)]
    assert len(refusals) / len(sample) == pytest.approx(0.25, abs=0.07)
    names = defaultdict(set)
    utterances = set()
    for row, spec in refusals:
        expected = spec["expected"]
        target = expected["unavailable"]
        domain, name = target["domain"], target["name"]
        assert domain in DOMAINS
        assert expected == expected_no_action(
            "refuse", subfamily="unexposed_domain", unavailable=target
        )
        assert gold_matches_family(expected, "absence")
        assert validate_spec(spec) is None
        assert name not in {entity["name"] for entity in spec["home"]["entities"]}
        assert {"names": name, "domain": domain} in exposed_entities(spec["home"])
        messages = row["messages"]
        assert not any(message.get("tool_calls") for message in messages)
        assert not any(message["role"] == "tool" for message in messages)
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == f"I can't control {name}. It is only available as context."
        assert is_refusal(messages[-1]["content"])
        assert serialize_context(spec["home"], spec["utterance"]) == messages[0]["content"]
        assert name in messages[0]["content"]
        user = messages[1]["content"]
        assert name.casefold() in user.casefold()
        assert not check_quality_eval_overlap(user)
        assert row["metadata"]["subcategory"] == "unexposed_domain"
        names[domain].add(name)
        utterances.add(user.casefold())

    assert set(names) == DOMAINS
    assert len(names["person"]) >= 3
    assert names["sun"] == {"Sun"}
    assert len(utterances) > 64

    ordinary = [spec for _, spec in sample if not _is_unexposed(spec)]
    assert ordinary
    assert all(
        spec["expected"]["response"] in {"device_absent", "area_unavailable"}
        for spec in ordinary
    )


def test_expected_unexposed_domain_targets_context_only_entities():
    rng = random.Random(7)
    for domain, name in (
        ("person", "Priya Patel"),
        ("sun", "Sun"),
    ):
        home = {
            "entities": [{"domain": "light", "name": "Desk Lamp"}],
            "context_entities": [{"domain": domain, "name": name}],
        }
        expected = expected_unexposed_domain(home, rng)
        assert expected == expected_no_action(
            "refuse", subfamily="unexposed_domain", unavailable={"domain": domain, "name": name}
        )
        assert gold_matches_family(expected, "absence")
    assert expected_unexposed_domain(
        {"entities": [], "context_entities": [{"domain": "sensor", "name": "Date"}]}, rng
    ) is None
    assert expected_unexposed_domain({"entities": []}, rng) is None
