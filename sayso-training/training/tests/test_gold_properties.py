
from __future__ import annotations

import random
import string

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from generators.capability_registry import CAPABILITIES, required_features
from generators.gold import gold_from_scenario
from generators.homes import make_entity
from generators.hypothesis_strategies import trainable_pairs, trainable_scenario_strategy

_WORD = st.text(alphabet=string.ascii_lowercase, min_size=4, max_size=10)
_OTHER_AREAS = ("Garage", "Attic", "Basement", "Shed")
_NAMED_PAIRS = tuple(
    pair for pair in trainable_pairs() if pair[0] not in {"timers", "scripts"}
)
_NAMED_CAPABILITIES = tuple(sorted({capability for capability, _ in _NAMED_PAIRS}))


def _named_calls(label: dict) -> list[str]:
    return [
        call["arguments"]["name"]
        for call in label.get("calls") or []
        if "name" in (call.get("arguments") or {})
    ]


def _with_name(label: dict, old: str, new: str) -> dict:
    calls = []
    for call in label.get("calls") or []:
        arguments = call.get("arguments") or {}
        if arguments.get("name") == old:
            call = {**call, "arguments": {**arguments, "name": new}}
        calls.append(call)
    return {**label, "calls": calls}


@given(scenario=trainable_scenario_strategy(_NAMED_PAIRS), token=_WORD)
@settings(max_examples=40, deadline=None)
def test_renaming_entity_renames_call_target(scenario: dict, token: str) -> None:
    target = scenario["target_entity"]
    assume(target is not None)
    old = target["name"]
    new = f"Renamed {token}"
    renamed = {**target, "name": new}
    entities = [
        renamed if entity["entity_id"] == target["entity_id"] else entity
        for entity in scenario["home"]["entities"]
    ]
    home = {**scenario["home"], "entities": entities}

    before = gold_from_scenario(scenario, random.Random(0))
    after = gold_from_scenario(
        {**scenario, "home": home, "target_entity": renamed}, random.Random(0)
    )

    assert after == _with_name(before, old, new)
    assert _named_calls(after) == [new]


@given(
    scenario=trainable_scenario_strategy(_NAMED_PAIRS),
    token=_WORD,
    area=st.sampled_from(_OTHER_AREAS),
    capability=st.sampled_from(_NAMED_CAPABILITIES),
)
@settings(max_examples=40, deadline=None)
def test_unrelated_entity_leaves_label_unchanged(
    scenario: dict, token: str, area: str, capability: str
) -> None:
    target = scenario["target_entity"]
    assume(target is not None)
    assume(area != target["area"])
    assume(CAPABILITIES[capability].domain != target["domain"])
    extra = make_entity(
        name=f"Unrelated {token}",
        capability=capability,
        area=area,
        floor="Main Floor",
        rng=random.Random(0),
    )
    home = {**scenario["home"], "entities": [*scenario["home"]["entities"], extra]}

    before = gold_from_scenario(scenario, random.Random(0))
    after = gold_from_scenario({**scenario, "home": home}, random.Random(0))

    assert after == before


_DISTRACTOR = "Pantry Shelf"


@given(pair=st.sampled_from(_NAMED_PAIRS), word=_WORD, alias=_WORD)
@settings(max_examples=60, deadline=None)
def test_alias_request_resolves_to_canonical_entity(
    pair: tuple[str, str], word: str, alias: str
) -> None:
    capability, operation = pair
    assume(alias != word)
    assume(alias not in _DISTRACTOR.casefold().split())
    assume(word not in _DISTRACTOR.casefold().split())
    canonical = f"Kitchen {word}"
    features = required_features(capability, operation)
    entity = make_entity(
        name=canonical,
        capability=capability,
        area="Kitchen",
        floor="Main Floor",
        rng=random.Random(0),
        aliases=[alias],
        features=features,
    )
    distractor = make_entity(
        name=_DISTRACTOR,
        capability=capability,
        area="Kitchen",
        floor="Main Floor",
        rng=random.Random(1),
        features=features,
    )
    home = {"sayso_entity_area": "Kitchen", "entities": [entity, distractor]}

    def request(name: str) -> dict:
        return {
            "home": home,
            "capability": capability,
            "operation": operation,
            "targeting": "individual",
            "robustness": "ambiguity",
            "request_intent": {"name": name},
        }

    via_alias = gold_from_scenario(request(alias), random.Random(0))
    via_canonical = gold_from_scenario(request(canonical), random.Random(0))

    assert via_alias == via_canonical
    assert via_alias["kind"] == "action"
    assert _named_calls(via_alias) == [canonical]
