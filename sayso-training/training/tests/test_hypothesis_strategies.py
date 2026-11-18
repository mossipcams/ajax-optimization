
from __future__ import annotations

from hypothesis import given, settings

from generators.capability_registry import CAPABILITIES, entity_supports, trainable_operations
from generators.hypothesis_strategies import (
    capability_operation_strategy,
    corrupted_spec_strategy,
    home_strategy,
    invalid_brightness_strategy,
    scenario_strategy,
    valid_brightness_strategy,
    valid_spec_strategy,
)
from generators.validation import validate_spec


@given(home=home_strategy())
@settings(max_examples=20, deadline=None)
def test_strategy_home_produces_entities(home: dict) -> None:
    assert len(home["entities"]) >= 8


@given(brightness=valid_brightness_strategy())
@settings(max_examples=40, deadline=None)
def test_valid_brightness_in_bounds(brightness: int) -> None:
    assert 0 <= brightness <= 100


@given(brightness=invalid_brightness_strategy())
@settings(max_examples=40, deadline=None)
def test_invalid_brightness_out_of_bounds(brightness: int) -> None:
    assert brightness not in range(0, 101)


@given(pair=capability_operation_strategy())
@settings(max_examples=40, deadline=None)
def test_capability_operation_pairs(pair: tuple[str, str]) -> None:
    cap, op = pair
    assert cap and op


@given(spec=valid_spec_strategy())
@settings(max_examples=40, deadline=None)
def test_valid_specs_pass_validation(spec: dict) -> None:
    assert validate_spec(spec) is None


@given(spec=valid_spec_strategy())
@settings(max_examples=40, deadline=None)
def test_valid_spec_target_supports_operation(spec: dict) -> None:
    assert spec["expected"]["kind"] == "action"
    entities = {entity["name"]: entity for entity in spec["home"]["entities"]}
    for call in spec["expected"]["calls"]:
        name = (call.get("arguments") or {}).get("name")
        if name in entities:
            assert entity_supports(entities[name], spec["capability"], spec["subcategory"])


def test_valid_spec_strategy_covers_and_validates_trainable_pairs() -> None:
    trainable = {
        (name, op.name) for name, cap in CAPABILITIES.items() for op in trainable_operations(cap)
    }
    covered: set[tuple[str, str]] = set()

    @given(spec=valid_spec_strategy())
    @settings(max_examples=2000, deadline=None)
    def collect(spec: dict) -> None:
        covered.add((spec["capability"], spec["subcategory"]))
        assert validate_spec(spec) is None

    collect()
    assert covered == trainable


@given(spec=corrupted_spec_strategy("wrong_entity"))
@settings(max_examples=20, deadline=None)
def test_corrupted_entity_specs_fail_validation(spec: dict) -> None:
    assert validate_spec(spec) == "unknown_canonical_entity"


@given(spec=corrupted_spec_strategy("wrong_tool"))
@settings(max_examples=20, deadline=None)
def test_corrupted_tool_specs_fail_validation(spec: dict) -> None:
    assert validate_spec(spec) is not None


@given(scenario=scenario_strategy(capability="lights", operation="turn_on"))
@settings(max_examples=20, deadline=None)
def test_scenario_semantic_id_ignores_attempt(scenario: dict) -> None:
    from generators.scenarios import semantic_id

    base = semantic_id(scenario)
    mutated = dict(scenario)
    mutated["attempt"] = scenario.get("attempt", 0) + 99
    mutated["scenario_index"] = scenario.get("scenario_index", 0) + 99
    assert semantic_id(mutated) == base
