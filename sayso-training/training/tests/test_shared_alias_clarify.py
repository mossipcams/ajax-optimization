from __future__ import annotations

import random
from functools import lru_cache

from evals.scorer import is_clarification
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.homes import _eval_entity_names
from generators.rendering import _final_text
from generators.row_generation import generate_row
from generators.shared_alias import HOME_SIZES, SHARED_ALIASES, build_shared_alias_spec
from generators.validation import shared_alias_action_conflict, validate_row

SEED = 7
SAMPLE = 400


def _slot(index):
    return {
        "index": index, "family": "clarify", "robustness": "ambiguity", "capability": "lights",
        "operation": "turn_on", "tier": 1, "home_size": HOME_SIZES[index % len(HOME_SIZES)],
    }


@lru_cache(maxsize=1)
def _rows():
    config = GeneratorConfig(count=SAMPLE, seed=SEED)
    tracker = DuplicateTracker()
    rows = []
    for index in range(SAMPLE):
        row, _ = generate_row(_slot(index), config, random.Random(index),
                              excluded=set(), dup_tracker=tracker)
        if row is not None:
            rows.append(row)
    return tuple(rows)


def _shared_rows():
    return [row for row in _rows() if row["metadata"].get("shared_alias")]


def _built_specs(count=400):
    return [spec for index in range(count)
            if (spec := build_shared_alias_spec(random.Random(index), index,
                                                HOME_SIZES[index % len(HOME_SIZES)])) is not None]


def _lights_specs():
    return [spec for spec in _built_specs() if spec["capability"] == "lights"]


def _scripts_specs():
    return [spec for spec in _built_specs() if spec["capability"] == "scripts"]


def test_shared_alias_clarify_rows_are_generated():
    assert len(_shared_rows()) >= 10


def test_script_variant_rows_are_generated_and_clarify():
    rows = [row for row in _shared_rows() if row["metadata"]["capability"] == "scripts"]
    assert rows
    for row in rows:
        reply = row["messages"][-1]
        assert not reply.get("tool_calls")
        assert is_clarification(reply["content"])
        assert all(f"the {name}" in reply["content"] for name in row["metadata"]["shared_candidates"])


def test_shared_alias_rows_are_clarifications_naming_both_entities():
    for row in _shared_rows():
        messages = row["messages"]
        user = next(m for m in reversed(messages) if m["role"] == "user")["content"]
        reply = messages[-1]
        alias = row["metadata"]["shared_alias"]
        candidates = row["metadata"]["shared_candidates"]
        assert alias in user.casefold()
        assert not reply.get("tool_calls")
        assert is_clarification(reply["content"])
        assert len(candidates) >= 2
        assert all(f"the {name}" in reply["content"] for name in candidates)


def test_built_lights_homes_inject_shared_alias_into_a_normal_home():
    specs = _lights_specs()
    assert len(specs) >= 10
    evals = _eval_entity_names()
    for spec in specs:
        alias = spec["shared_alias"]
        entities = spec["home"]["entities"]
        candidates = set(spec["expected"]["candidates"])
        group = [entity for entity in entities if entity["name"] in candidates]
        assert 2 <= len(group) <= 3
        assert len(entities) >= 8
        assert len(entities) > len(group)
        assert len(spec["home"]["entities"]) == spec["home"]["size"]
        assert len({entity["area"] for entity in group}) == 1
        assert group[0]["area"] in spec["home"]["areas"]
        assert alias in SHARED_ALIASES
        assert alias not in {"reading light", "bedside lamp"}
        names = [entity["name"].casefold() for entity in entities]
        assert len(set(names)) == len(names)
        assert not set(names) & evals
        for entity in group:
            assert alias in {name.casefold() for name in entity["aliases"]}
            assert alias not in entity["name"].casefold()
        for entity in entities:
            if entity["name"] not in candidates:
                assert alias not in {name.casefold() for name in [entity["name"], *entity["aliases"]]}
        assert alias in spec["utterance"].casefold()
        reply = _final_text(spec)
        assert is_clarification(reply)
        assert all(f"the {name}" in reply for name in candidates)


def test_script_variant_clarifies_between_routines():
    specs = _scripts_specs()
    assert specs
    assert 0.05 <= len(specs) / len(_built_specs()) <= 0.3
    for spec in specs:
        scripts = [entity for entity in spec["home"]["entities"] if entity["capability"] == "scripts"]
        names = {entity["name"] for entity in scripts}
        assert 2 <= len(scripts) <= 3
        assert set(spec["expected"]["candidates"]) == names
        assert not names & {"Evening Cleanup", "Welcome Home"}
        assert spec["shared_alias"] in spec["utterance"].casefold()
        assert is_clarification(_final_text(spec))


def test_shared_alias_action_is_rejected_and_plain_name_action_is_allowed():
    for spec in _lights_specs():
        target = next(item for item in spec["home"]["entities"] if item["name"] in spec["expected"]["candidates"])
        action = {
            **spec,
            "expected": {"kind": "action", "calls": [{"name": "HassTurnOn", "arguments": {
                "name": target["name"], "domain": ["light"]}}]},
        }
        assert shared_alias_action_conflict(action)
        assert validate_row(action) == "shared_alias_action"
        named = {**action, "utterance": f"turn on the {target['name']}"}
        assert not shared_alias_action_conflict(named)
