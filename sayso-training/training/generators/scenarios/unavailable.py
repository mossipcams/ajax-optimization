
from __future__ import annotations

import random
from typing import Any

from generators.gold import target_names_from_expected
from generators.homes import _ENTITY_TEMPLATES, make_entity
from generators.tools import build_call_for_operation
from generators.utterances import request_seed_from_spec

_UNAVAILABLE_REQUESTS: dict[tuple[str, str], str | tuple[str, ...]] = {
    ("lawn_mowers", "control"): "start mowing the lawn with {name}",
    ("todo_lists", "control"): "add milk to {name}",
    ("buttons", "control"): "press {name}",
    ("covers", "set_position"): (
        "set {name} to 40 percent open", "set {name} halfway", "set {name} to 30 percent",
    ),
    ("media_players", "navigate"): ("go home on {name}", "press back on {name}"),
    ("media_players", "launch_app"): ("open Netflix on {name}", "open YouTube on {name}"),
    ("scripts", "query_state"): "what is the status of {name}",
}


def unique_no_action_hint(spec: dict[str, Any], rng: random.Random) -> str:
    requested = spec["expected"].get("requested")
    if requested:
        return request_seed_from_spec({
            **spec, "expected": requested,
            "target_names": target_names_from_expected(requested),
        })
    capability = spec["capability"]
    operation = spec["operation"]
    intent = spec.get("request_intent") or {}
    area = intent.get("area") or spec["home"]["sayso_entity_area"]
    entity = None
    if capability != "timers":
        noun = intent.get("name") or ("routine" if capability == "scripts" else _ENTITY_TEMPLATES[capability][0].lower())
        entity = make_entity(
            name=f"the {area} {noun}",
            capability=capability, area=area, floor="Main Floor", rng=rng,
        )
    template = _UNAVAILABLE_REQUESTS.get((capability, operation))
    if template:
        if isinstance(template, tuple):
            template = rng.choice(template)
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": f"{capability}.{operation}"}]
        return template.format(name=entity["name"] if entity else "it")
    call = build_call_for_operation(entity, capability, operation, rng, area=area)
    spec["hint_call"] = call
    return request_seed_from_spec({
        "expected": {"kind": "action", "calls": [call]},
        "target_names": [entity["name"]] if entity else [""],
        "phrasing_seed": spec.get("phrasing_seed"),
    })
