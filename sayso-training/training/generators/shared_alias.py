from __future__ import annotations

import random
from typing import Any

from generators.capability_registry import CAPABILITIES
from generators.gold import expected_no_action
from generators.homes import _eval_entity_names, generate_home, make_entity

SHARED_ALIAS_RATE = 0.6
SCRIPT_VARIANT_SHARE = 0.15
HOME_SIZES: tuple[int, ...] = (8, 16, 32, 64)

SHARED_ALIASES: tuple[str, ...] = (
    "desk lamp", "side lamp", "floor lamp", "corner lamp", "night light", "ceiling light",
    "wall light", "overhead light", "work light", "task light", "mood light", "accent light",
    "ambient light", "porch light", "main light", "soft light", "counter light", "cove light",
    "strip light", "lounge light",
)
_LIGHT_NOUNS: tuple[str, ...] = ("Pendant", "Spot", "Strip", "Track", "Sconce", "Panel", "Lamp", "Fixture")
_FIRST_NAMES: tuple[str, ...] = (
    "Priya", "Mateo", "Noor", "Ellis", "Hana", "Tomas", "Ivy", "Dev", "Leona", "Oscar", "Wren", "Yusuf",
)
_ROUTINE_NAMES: tuple[str, ...] = (
    "Bedtime Routine", "Morning Routine", "Movie Mode", "Dinner Party",
    "Away Mode", "Good Morning", "Nap Time", "Study Session",
)
_SCRIPT_UTTERANCES: tuple[tuple[str, str], ...] = (
    ("run the routine", "routine"),
    ("start the script", "script"),
    ("kick off my routine please", "routine"),
    ("run my routine", "routine"),
    ("start the routine", "routine"),
    ("run the script please", "script"),
)
_TEMPLATES: dict[str, tuple[str, ...]] = {
    "turn_on": (
        "switch on the {alias}",
        "turn on the {alias} in the {room}",
        "could you turn on the {alias}?",
        "can you switch on the {alias} in the {room} for me",
        "turn the {alias} on",
    ),
    "turn_off": (
        "switch the {alias} off",
        "turn off the {alias} in the {room}",
        "please switch off the {alias} in the {room}",
        "turn the {alias} off",
    ),
    "set_brightness": (
        "dim the {alias} to {pct} percent",
        "set the {alias} in the {room} to {pct} percent",
        "make the {alias} a little brighter",
        "make the {alias} in the {room} a little dimmer",
        "brighten the {alias} in the {room}",
    ),
    "set_color": (
        "set the {alias} to warm white",
        "make the {alias} in the {room} blue",
        "change the {alias} to green",
        "set the {alias} in the {room} to purple",
    ),
    "set_color_temperature": (
        "set the {alias} in the {room} to {kelvin} kelvin",
        "make the {alias} warmer",
        "make the {alias} in the {room} cooler",
        "set the {alias} to {kelvin} kelvin",
    ),
}
_PERCENTS = (20, 25, 30, 40, 60, 75)
_KELVIN = (2700, 3000, 4000, 5000, 6500)


def _slug(text: str) -> str:
    return "_".join(text.casefold().split())


def _texts(home: dict[str, Any]) -> set[str]:
    return {
        text.casefold()
        for item in home["entities"]
        for text in (item["name"], *item.get("aliases", []))
    } | {_slug(item["name"]) for item in home["entities"]}


def _claim(name: str, alias: str, taken: set[str], evals: frozenset[str]) -> bool:
    key = name.casefold()
    if key in taken or key in evals or _slug(name) in taken or (alias and alias in key):
        return False
    taken.update({key, _slug(name)})
    return True


def _lights_name(rng: random.Random, area: str, home: dict[str, Any]) -> str:
    noun = rng.choice(_LIGHT_NOUNS)
    if rng.random() < 0.4:
        prefix = rng.choice([*home.get("owners", ()), *(f"{first}'s" for first in _FIRST_NAMES)])
        return f"{prefix} {noun}"
    return f"{area} {noun}"


def _spec(
    index: int,
    home: dict[str, Any],
    candidates: list[str],
    utterance: str,
    alias: str,
    capability: str,
    operation: str,
) -> dict[str, Any]:
    return {
        "candidate_id": f"shared_alias_{index:06d}",
        "seed": index,
        "category": "ambiguity",
        "subcategory": "shared_alias",
        "family": "clarify",
        "home": home,
        "expected": expected_no_action("clarify", candidates=candidates),
        "target_names": [],
        "spoken_targets": {},
        "excluded_names": [],
        "exclusion_scope": None,
        "contrastive_group": None,
        "phrasing_seed": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": utterance,
        "capability": capability,
        "operation": operation,
        "tier": CAPABILITIES[capability].tier,
        "semantic_id": f"shared_alias_{index:06d}_{_slug(alias)}",
        "removed_tools": [],
        "full_tool_catalog": True,
        "shared_alias": alias,
    }


def _light_spec(rng: random.Random, index: int, home: dict[str, Any], evals: frozenset[str]) -> dict[str, Any] | None:
    taken = _texts(home)
    options = [
        candidate for candidate in SHARED_ALIASES
        if candidate not in evals and not any(candidate in text for text in taken)
    ]
    if not options:
        return None
    alias = rng.choice(options)
    area = rng.choice(home["areas"])
    floor = home["area_floors"][area]
    wanted = 3 if rng.random() < 0.3 else 2
    names: list[str] = []
    for _ in range(64):
        if len(names) == wanted:
            break
        name = _lights_name(rng, area, home)
        if _claim(name, alias, taken, evals):
            names.append(name)
    if len(names) < 2:
        return None
    home["entities"].extend(
        make_entity(name=name, capability="lights", area=area, floor=floor, rng=rng,
                    aliases=[name, alias], state="off")
        for name in names
    )
    home["size"] = len(home["entities"])
    operation = rng.choice(tuple(_TEMPLATES))
    template = rng.choice(_TEMPLATES[operation])
    utterance = template.format(
        alias=alias, room=area.casefold(), pct=rng.choice(_PERCENTS), kelvin=rng.choice(_KELVIN),
    )
    candidates = list(names)
    rng.shuffle(candidates)
    return _spec(index, home, candidates, utterance, alias, "lights", operation)


def _script_spec(rng: random.Random, index: int, home: dict[str, Any], evals: frozenset[str]) -> dict[str, Any] | None:
    taken = _texts(home)
    scripts = [item for item in home["entities"] if item["capability"] == "scripts"]
    if len(scripts) > 3:
        return None
    needed = max(0, 2 - len(scripts))
    pool = [name for name in _ROUTINE_NAMES if name.casefold() not in evals]
    rng.shuffle(pool)
    area = rng.choice(home["areas"])
    added: list[dict[str, Any]] = []
    for name in pool:
        if len(added) == needed:
            break
        if _claim(name, "", taken, evals):
            added.append(make_entity(name=name, capability="scripts", area=area,
                                     floor=home["area_floors"][area], rng=rng, aliases=[name]))
    if len(added) < needed:
        return None
    home["entities"].extend(added)
    home["size"] = len(home["entities"])
    candidates = [item["name"] for item in home["entities"] if item["capability"] == "scripts"]
    rng.shuffle(candidates)
    utterance, alias = rng.choice(_SCRIPT_UTTERANCES)
    return _spec(index, home, candidates, utterance, alias, "scripts", "run")


def build_shared_alias_spec(rng: random.Random, index: int, home_size: int = 16) -> dict[str, Any] | None:
    evals = _eval_entity_names()
    home = generate_home(index, home_size, random.Random(f"shared_alias_home:{index}:{home_size}"), holdout=False)
    if rng.random() < SCRIPT_VARIANT_SHARE:
        spec = _script_spec(rng, index, home, evals)
        if spec is not None:
            return spec
    return _light_spec(rng, index, home, evals)
