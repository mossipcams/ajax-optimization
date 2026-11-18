
from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Any

from generators.homes import entities_in_area, make_entity


def entity(
    name: str,
    capability: str,
    area: str,
    *,
    floor: str = "Main Floor",
    aliases: list[str] | None = None,
    features: tuple[str, ...] | None = None,
    entity_id: str | None = None,
    device_class: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    made = make_entity(
        name=name,
        capability=capability,
        area=area,
        floor=floor,
        rng=random.Random(0),
        aliases=aliases or [name],
        features=features,
        state=state or "off",
        device_class=device_class,
    )
    if entity_id:
        made["entity_id"] = entity_id
    return made


def home(home_id: str, entities: list[dict[str, Any]], *, sayso_entity_area: str) -> dict[str, Any]:
    floors = {}
    for item in entities:
        floors.setdefault(item["area"], item.get("floor") or "Main Floor")
    floors.setdefault(sayso_entity_area, "Main Floor")
    return {
        "home_id": home_id,
        "size": len(entities),
        "sayso_entity_area": sayso_entity_area,
        "entities": entities,
        "active_timers": [],
        "areas": list(floors),
        "area_floors": floors,
        "owners": (),
        "exposure_source": "synthetic",
    }


def variant(
    *,
    prefix: str,
    label: str,
    capability: str,
    operation: str,
    targeting: str,
    robustness: str,
    entities: list[dict[str, Any]],
    area: str,
    inject_missing: bool = True,
    utterance: str | None = None,
    rng_index: int | None = None,
    request_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "family": f"{prefix}_{label}",
        "phrasing_seed": prefix,
        "capability": capability,
        "operation": operation,
        "targeting": targeting,
        "robustness": robustness,
        "inject_missing": inject_missing,
        "utterance": utterance,
        "rng_index": rng_index,
        "request_intent": request_intent,
        "home": home(f"{prefix}_{label}", entities, sayso_entity_area=area),
    }


def _distractors(area: str, elsewhere: str) -> list[dict[str, Any]]:
    return [
        entity("Floor Lamp", "lights", area),
        entity("Corner Lamp", "lights", area),
        entity("Console Outlet", "switches", area),
        entity(f"{elsewhere} Speaker", "media_players", elsewhere,
               features=("play", "pause", "volume", "volume_step", "mute", "next", "previous")),
    ]


def media_presence_family(
    *,
    prefix: str,
    target_name: str,
    renamed: str,
    area: str,
    elsewhere: str,
    entity_id: str,
    aliases: list[str] | None = None,
) -> list[dict[str, Any]]:
    features = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")

    def player(name: str, in_area: str) -> dict[str, Any]:
        return entity(name, "media_players", in_area, aliases=aliases or [name],
                      features=features, entity_id=entity_id, device_class="tv")

    common = dict(
        prefix=prefix, capability="media_players", operation="turn_on",
        targeting="area", robustness="ambiguity", area=area,
        utterance=f"Turn on the {area.casefold()} media player",
    )
    return [
        variant(label="present", entities=[player(target_name, area)], **common),
        variant(label="renamed", entities=[player(renamed, area)], **common),
        variant(
            label="moved",
            entities=[player(target_name, elsewhere), entity("Reading Lamp", "lights", area)],
            inject_missing=False,
            **common,
        ),
        variant(
            label="distractors",
            entities=[player(target_name, area), *_distractors(area, elsewhere)],
            **common,
        ),
    ]


_SCREEN_FEATURES = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")


def _screen(name: str, area: str, *, aliases: list[str] | None = None,
            entity_id: str | None = None) -> dict[str, Any]:
    return entity(name, "media_players", area, aliases=aliases or [name],
                  features=_SCREEN_FEATURES, entity_id=entity_id, device_class="tv")


def _speaker(name: str, area: str) -> dict[str, Any]:
    return entity(name, "media_players", area, device_class="speaker",
                  features=("on", "off", "play", "pause", "volume", "volume_step", "mute"))


def named_device_family(
    *,
    prefix: str,
    area: str,
    elsewhere: str,
    device: str,
    entity_id: str,
    operation: str = "turn_on",
    utterance: str,
    from_elsewhere: bool = False,
) -> list[dict[str, Any]]:
    satellite_area = elsewhere if from_elsewhere else area
    names_room = area.casefold() in utterance.casefold()
    common = dict(
        prefix=prefix, capability="media_players", operation=operation,
        targeting="individual", robustness="ambiguity", area=satellite_area,
        utterance=utterance, rng_index=0,
        request_intent={"name": device, "area": area if names_room else None},
    )
    return [
        variant(label="bare", entities=[_screen(device, area)], **common),
        variant(label="room_named", entities=[_screen(f"{area} {device}", area)], **common),
        variant(
            label="aliased",
            entities=[_screen("Screen One", area, aliases=["Screen One", device],
                              entity_id=entity_id)],
            **common,
        ),
        variant(
            label="speaker",
            entities=[_screen(device, area, entity_id=entity_id),
                      _speaker(f"{area} Media Player", area)],
            **common,
        ),
        variant(
            label="absent",
            entities=[_speaker(f"{area} Media Player", area),
                      _screen(device, elsewhere),
                      entity("Reading Lamp", "lights", area)],
            inject_missing=False,
            **common,
        ),
        variant(
            label="ambiguous",
            entities=[_screen(f"{area} {device}", area), _screen(f"Second {device}", area)],
            **common,
        ),
        variant(
            label="irrelevant",
            entities=[_screen(device, area), entity("Floor Lamp", "lights", area),
                      entity("Hall Switch", "switches", elsewhere)],
            **common,
        ),
    ]


def ambiguity_family(*, prefix: str, area: str, names: tuple[str, str]) -> list[dict[str, Any]]:
    first, second = names
    common = dict(
        prefix=prefix, capability="lights", operation="turn_on",
        targeting="area", robustness="ambiguity", area=area,
    )
    return [
        variant(label="single", entities=[entity(first, "lights", area)], **common),
        variant(
            label="pair",
            entities=[entity(first, "lights", area), entity(second, "lights", area)],
            **common,
        ),
    ]


def domain_family(*, prefix: str, area: str, fan_name: str, light_name: str) -> list[dict[str, Any]]:
    common = dict(
        prefix=prefix, capability="fans", operation="turn_on",
        targeting="area", robustness="ambiguity", area=area,
    )
    return [
        variant(
            label="present",
            entities=[entity(fan_name, "fans", area), entity(light_name, "lights", area)],
            **common,
        ),
        variant(
            label="absent",
            entities=[entity(light_name, "lights", area)],
            inject_missing=False,
            **common,
        ),
    ]


def supported_action_family(*, prefix: str, area: str, name: str) -> list[dict[str, Any]]:
    common = dict(
        prefix=prefix, capability="media_players", operation="volume_set",
        targeting="area", robustness="ambiguity", area=area, rng_index=0,
    )
    return [
        variant(
            label="capable",
            entities=[entity(name, "media_players", area, device_class="speaker",
                             features=("on", "off", "play", "pause", "volume", "volume_step", "mute"))],
            **common,
        ),
    ]


def alias_family(*, prefix: str, area: str, name: str, alias: str) -> list[dict[str, Any]]:
    return [
        variant(
            prefix=prefix, label="alias", capability="lights", operation="turn_on",
            targeting="individual", robustness="alias_distractor", area=area,
            rng_index=0,
            entities=[entity(name, "lights", area, aliases=[name, alias]),
                      entity("Ceiling Light", "lights", area)],
        ),
    ]


def canonical_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    variants += media_presence_family(
        prefix="ground_media_a",
        target_name="Den Projector",
        renamed="Beamer",
        area="Den",
        elsewhere="Attic",
        entity_id="media_player.den_projector",
    )
    variants += media_presence_family(
        prefix="ground_media_b",
        target_name="Sunroom Screen",
        renamed="Panorama",
        area="Sunroom",
        elsewhere="Workshop",
        entity_id="media_player.sunroom_screen",
        aliases=["Sunroom Screen", "the screen"],
    )
    variants += ambiguity_family(
        prefix="ground_light", area="Craft Room", names=("Bench Lamp", "Shelf Lamp")
    )
    variants += domain_family(
        prefix="ground_domain", area="Pantry", fan_name="Pantry Extractor", light_name="Pantry Strip"
    )
    variants += supported_action_family(
        prefix="ground_features", area="Landing", name="Landing Speaker"
    )
    variants += alias_family(
        prefix="ground_alias", area="Music Room", name="Practice Lamp", alias="rehearsal light"
    )
    variants += named_device_family(
        prefix="ground_named",
        area="Loft",
        elsewhere="Cellar",
        device="Television",
        entity_id="media_player.loft_television",
        operation="turn_off",
        utterance="Turn off the television",
    )
    return variants


_RESERVED_AREAS: frozenset[str] = frozenset({
    "Den", "Attic", "Sunroom", "Workshop", "Craft Room", "Pantry", "Landing", "Music Room",
    "Loft", "Cellar",
    "Living Room", "Bedroom", "Office", "Basement", "Kitchen", "Bathroom", "Porch",
})

_RESERVED_DEVICE_WORDS = frozenset({"TV"})
_SITE_SCREENS = tuple(
    word for word in ("TV", "Television", "OLED TV", "Streaming TV", "Display", "Projector")
    if word not in _RESERVED_DEVICE_WORDS
)
_SITE_RENAMES = ("Beamer", "Panorama", "Big Picture", "The Set", "Screen One", "Viewer")
_SITE_LAMPS = ("Bench Lamp", "Shelf Lamp", "Desk Lamp", "Corner Lamp", "Floor Lamp",
               "Table Lamp", "Wall Light", "Pendant Light", "Ceiling Lights", "Strip Light")
_SITE_FANS = ("Extractor", "Ceiling Fan", "Tower Fan", "Desk Fan")
_SITE_ALIASES = ("accent light", "task light", "side light", "spare light", "nook light")


def _slug(text: str) -> str:
    return "_".join(part for part in re.split(r"[^a-z0-9]+", text.casefold()) if part)


def site_variants(index: int, area: str, elsewhere: str) -> list[dict[str, Any]]:
    tag = f"s{index:02d}"
    screen = _SITE_SCREENS[index % len(_SITE_SCREENS)]
    lamp_a = _SITE_LAMPS[index % len(_SITE_LAMPS)]
    lamp_b = _SITE_LAMPS[(index + 3) % len(_SITE_LAMPS)]
    lamp_c = _SITE_LAMPS[(index + 6) % len(_SITE_LAMPS)]
    variants: list[dict[str, Any]] = []
    variants += media_presence_family(
        prefix=f"ground_media_{tag}",
        target_name=f"{area} {screen}",
        renamed=_SITE_RENAMES[index % len(_SITE_RENAMES)],
        area=area,
        elsewhere=elsewhere,
        entity_id=f"media_player.{_slug(area)}_{_slug(screen)}_{tag}",
    )
    variants += ambiguity_family(
        prefix=f"ground_light_{tag}", area=area, names=(f"{area} {lamp_a}", f"{area} {lamp_b}")
    )
    variants += domain_family(
        prefix=f"ground_domain_{tag}",
        area=area,
        fan_name=f"{area} {_SITE_FANS[index % len(_SITE_FANS)]}",
        light_name=f"{area} {lamp_c}",
    )
    variants += supported_action_family(
        prefix=f"ground_features_{tag}", area=area, name=f"{area} Speaker"
    )
    variants += alias_family(
        prefix=f"ground_alias_{tag}",
        area=area,
        name=f"{area} {lamp_b}",
        alias=_SITE_ALIASES[index % len(_SITE_ALIASES)],
    )
    lowered = area.casefold()
    wording, operation, away = (
        ("bare", "turn_on", False),
        ("room", "turn_off", False),
        ("away", "turn_on", True),
    )[index % 3]
    utterance = {
        "bare": f"Turn on the {screen.casefold()}",
        "room": f"Turn off the {lowered} {screen.casefold()}",
        "away": f"Turn on the {screen.casefold()} in the {lowered}",
    }[wording]
    variants += named_device_family(
        prefix=f"ground_named_{wording}_{tag}", area=area, elsewhere=elsewhere,
        device=screen,
        entity_id=f"media_player.{_slug(area)}_{_slug(screen)}_{wording}_{tag}",
        operation=operation, utterance=utterance, from_elsewhere=away,
    )
    return variants


def _sites() -> list[tuple[str, str]]:
    from generators.homes import _AREAS

    areas = [area for area in _AREAS if area not in _RESERVED_AREAS]
    return [(area, areas[(i + 1) % len(areas)]) for i, area in enumerate(areas)]


@lru_cache(maxsize=1)
def _catalogue() -> tuple[dict[str, Any], ...]:
    variants = canonical_variants()
    for index, (area, elsewhere) in enumerate(_sites()):
        variants += site_variants(index, area, elsewhere)
    return tuple(variants)


def training_variants() -> list[dict[str, Any]]:
    return list(_catalogue())


def required_training_variants() -> list[dict[str, Any]]:
    canonical = {variant["family"] for variant in canonical_variants()}
    return [
        variant for variant in training_variants()
        if variant["family"] in canonical
        and not variant["family"].startswith("ground_media_b_")
        and variant["family"] not in {"ground_named_absent", "ground_named_ambiguous"}
    ]


def build_spec(variant: dict[str, Any], *, seed: int = 20260910, index: int = 0) -> dict[str, Any]:
    import copy

    from generators.labels import scenario_to_spec
    from generators.scenarios.unavailable import unique_no_action_hint
    from generators.scenarios import build_scenario
    from generators.utterances import apply_generic_wording, expand_utterance

    scenario = build_scenario(
        index=variant.get("rng_index") if variant.get("rng_index") is not None else index,
        seed=seed,
        capability=variant["capability"],
        operation=variant["operation"],
        home_size=len(variant["home"]["entities"]),
        targeting=variant["targeting"],
        robustness=variant["robustness"],
        home=copy.deepcopy(variant["home"]),
        inject_missing=variant.get("inject_missing", True),
        request_intent=variant.get("request_intent"),
    )
    if variant.get("withheld_tools"):
        scenario["expected"] = variant["expected"]
    scenario["phrasing_seed"] = variant["phrasing_seed"]
    spec = scenario_to_spec(scenario)
    spec["grounding_family"] = variant["family"]
    if spec["expected"].get("kind") == "no_action":
        requested = _requested_names(spec)
        if requested:
            spec["spoken_targets"] = apply_generic_wording(
                {**spec, "target_names": requested}
            )["spoken_targets"]
        spec["request_hint"] = unique_no_action_hint(spec, random.Random(seed))
    elif variant["robustness"] == "ambiguity":
        apply_generic_wording(spec)
    spec["utterance"] = variant.get("utterance") or expand_utterance(spec)
    return spec


def _requested_names(spec: dict[str, Any]) -> list[str]:
    from generators.gold import target_names_from_expected

    return target_names_from_expected(spec["expected"].get("requested") or {})


def build_row(variant: dict[str, Any], *, seed: int = 20260910, index: int = 0) -> dict[str, Any]:
    from generators.labels import render_example

    spec = build_spec(variant, seed=seed, index=index)
    row = render_example(spec)
    row["metadata"]["grounding_family"] = variant["family"]
    return row


def pick_variant(
    capability: str, operation: str, index: int, *, variants: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    pool = [
        variant
        for variant in (variants if variants is not None else training_variants())
        if variant["capability"] == capability and variant["operation"] == operation
    ]
    if not pool:
        return None
    return pool[index % len(pool)]


GROUNDING_FAMILY_SLACK = 2

GROUNDING_CARRIER_FAMILIES = frozenset({"ordinary", "absence", "unsupported", "clarify"})


@lru_cache(maxsize=None)
def carrier_variants(family: str) -> tuple[dict[str, Any], ...]:
    if family not in GROUNDING_CARRIER_FAMILIES:
        return ()
    from generators.gold import gold_matches_family

    return tuple(
        variant
        for variant in training_variants()
        if gold_matches_family(build_spec(variant)["expected"], family)
    )


def is_blocked_request(family: str, capability: Any, operation: Any) -> bool:
    from generators.planning import BLOCKED_REQUEST_OPERATIONS

    return family == "unsupported" and (capability, operation) in BLOCKED_REQUEST_OPERATIONS


def pick_carrier_variant(
    slot: dict[str, Any], rng: random.Random, missing: Any = (), exclude: Any = ()
) -> dict[str, Any] | None:
    family = slot.get("family") or ""
    # Blocked-request slots keep their own capability/operation and refusal wording; no catalogue
    # variant may overwrite them, and no blocked pair may leak into an ordinary unsupported slot.
    if is_blocked_request(family, slot.get("capability"), slot.get("operation")):
        return None
    pool = [
        v for v in carrier_variants(family)
        if v["family"] in missing and v["family"] not in exclude
        and not is_blocked_request(family, v["capability"], v["operation"])
    ]
    if slot.get("grounding"):
        return rng.choice(pool) if pool else None
    same_pair = [
        v for v in pool
        if (v["capability"], v["operation"]) == (slot.get("capability"), slot.get("operation"))
    ]
    return rng.choice(same_pair or pool) if pool else None


@lru_cache(maxsize=1)
def grounding_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        (variant["capability"], variant["operation"])
        for variant in required_training_variants()
    )


def slot_can_carry_grounding(slot: dict[str, Any] | None) -> bool:
    if not slot or slot.get("area_scenario"):
        return False
    return bool(carrier_variants(slot.get("family") or ""))


def grounding_capacity(config: Any) -> int:
    return len(training_variants()) * config.near_duplicate_limit


def slot_ceiling(config: Any, pairs: set[tuple[str, str]]) -> float:
    from generators.sampling import build_quota_plan

    plan = build_quota_plan(config.count, config.seed, config.tier_proportions)
    matching = sum(
        1 for slot in plan if (slot["capability"], slot["operation"]) in pairs
    )
    share = matching / max(len(plan), 1)
    if config.real_home_path and config.real_home_rate:
        share *= max(0.0, 1.0 - config.real_home_rate)
    return min(1.0, share)


def grounding_available_share(config: Any) -> float:
    catalogue_share = grounding_capacity(config) / max(config.count, 1)
    return min(slot_ceiling(config, grounding_pairs()), catalogue_share)


CONTRAST_AXES = ("presence", "domain", "count", "area", "feature", "distractor")
CONTRAST_GROUP_SIZE = 4
_COMMAND_RETRIES = 32
_GROUP_ATTEMPTS = 8


class ContrastCommandError(ValueError):
    """No phrasing of this base's command clears the eval-overlap checks."""


CONTRAST_OPERATIONS: dict[str, tuple[str, ...]] = {
    "lights": ("turn_on", "turn_off"),
    "fans": ("turn_on", "turn_off"),
    "switches": ("turn_on", "turn_off"),
    "media_players": ("turn_on", "turn_off", "pause", "mute", "next_track"),
}


def _clone(household: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(household)


def _find(household: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in household["entities"] if item["name"] == name)


def mutate_presence(household: dict[str, Any], target: str) -> dict[str, Any]:
    mutated = _clone(household)
    mutated["entities"] = [item for item in mutated["entities"] if item["name"] != target]
    return mutated


def mutate_domain(household: dict[str, Any], target: str, domain: str = "sensor") -> dict[str, Any]:
    removed = _find(household, target)
    mutated = mutate_presence(household, target)
    mutated["context_entities"] = [
        *(mutated.get("context_entities") or []),
        {"name": target, "domain": domain, "area": removed["area"]},
    ]
    return mutated


def mutate_count(household: dict[str, Any], target: str) -> dict[str, Any]:
    mutated = _clone(household)
    original = _find(mutated, target)
    twin = _clone(original)
    twin["name"] = f"{target} 2"
    twin["aliases"] = [twin["name"]]
    twin["entity_id"] = f"{original['entity_id']}_2"
    mutated["entities"].append(twin)
    return mutated


def mutate_area(household: dict[str, Any], target: str, area: str) -> dict[str, Any]:
    mutated = _clone(household)
    moved = _find(mutated, target)
    moved["area"] = area
    moved["floor"] = (mutated.get("area_floors") or {}).get(area, moved.get("floor"))
    return mutated


def mutate_feature(household: dict[str, Any], target: str, operation: str) -> dict[str, Any]:
    from generators.capability_registry import required_features

    mutated = _clone(household)
    stripped = _find(mutated, target)
    dropped = set(required_features(stripped["capability"], operation))
    stripped["capabilities"] = [item for item in stripped["capabilities"] if item not in dropped]
    return mutated


def mutate_distractor(household: dict[str, Any], target: str) -> dict[str, Any]:
    mutated = _clone(household)
    original = _find(mutated, target)
    suffix = "Active App" if original["capability"] == "media_players" else "Power"
    mutated["context_entities"] = [
        *(mutated.get("context_entities") or []),
        {"name": f"{target} {suffix}", "domain": "sensor", "area": original["area"]},
    ]
    return mutated


def contrast_gold(
    household: dict[str, Any], capability: str, operation: str, noun: str, *, seed: int = 0
) -> dict[str, Any]:
    from generators.gold import expected_no_action, gold_from_scenario, refers_to
    from generators.tools import build_call_for_operation
    from generators.coverage import bare_tool_name

    expected = gold_from_scenario(
        {
            "home": household,
            "capability": capability,
            "operation": operation,
            "targeting": "individual",
            "robustness": "ambiguity",
            "request_intent": {"name": noun, "area": None},
        },
        random.Random(seed),
    )
    removed = withheld_tools(household, capability, operation)
    if expected.get("kind") == "reject" and removed:
        target = next(
            item for item in household["entities"]
            if item["capability"] == capability
            and item["area"] == household["sayso_entity_area"]
            and refers_to(item, noun)
        )
        requested = {"kind": "action", "calls": [
            build_call_for_operation(target, capability, operation, random.Random(seed))
        ]}
        return expected_no_action(
            "unsupported", requested=requested,
            unavailable_tools=[bare_tool_name(tool) for tool in removed],
        )
    return expected


def behaviour_class(expected: dict[str, Any]) -> str:
    from generators.gold import gold_matches_family

    for family, label in (
        ("ordinary", "action"),
        ("clarify", "clarify"),
        ("absence", "no_action"),
        ("unsupported", "refusal"),
        ("unavailable", "refusal"),
    ):
        if gold_matches_family(expected, family):
            return label
    raise ValueError(f"no behaviour class for {expected}")


def model_visible_facts(household: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    facts: dict[str, tuple[Any, ...]] = {}
    for item in household["entities"]:
        facts[item["name"].casefold()] = (
            item["domain"], item.get("area"),
            tuple(sorted(dict.fromkeys(item.get("aliases") or []))),
            tuple(sorted(item.get("capabilities") or ())),
        )
    for item in household.get("context_entities") or []:
        facts[item["name"].casefold()] = (item["domain"], item.get("area"), (), ())
    return facts


def _noun(target: dict[str, Any]) -> str | None:
    name, area = target["name"], target["area"]
    if not name.casefold().startswith(f"{area.casefold()} "):
        return None
    return name[len(area):].strip()


def _contrast_bases(household: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    from generators.capability_registry import entity_supports, is_satellite
    from generators.gold import refers_to

    bases = []
    for target in household["entities"]:
        noun = _noun(target)
        if noun is None or is_satellite(target):
            continue
        same = [
            item for item in entities_in_area(household, target["capability"], target["area"])
            if refers_to(item, noun)
        ]
        if same != [target]:
            continue
        for operation in CONTRAST_OPERATIONS.get(target["capability"], ()):
            if entity_supports(target, target["capability"], operation):
                bases.append((target, noun, operation))
    return bases


def _withholdable_tool(capability: str, operation: str) -> str | None:
    from generators.capability_registry import operation_spec, required_features
    from generators.planning import CORE_TOOL_NAMES

    op = operation_spec(capability, operation)
    if op is None or not op.tool_name or op.tool_name in CORE_TOOL_NAMES:
        return None
    return op.tool_name if required_features(capability, operation) else None


def withheld_tools(household: dict[str, Any], capability: str, operation: str) -> list[str]:
    from generators.capability_registry import entity_supports
    from generators.tools import namespaced_tool_name

    tool = _withholdable_tool(capability, operation)
    if tool is None or any(
        entity_supports(item, capability, operation)
        for item in household["entities"] if item["capability"] == capability
    ):
        return []
    return [namespaced_tool_name(tool)]


def _axes_for(household: dict[str, Any], target: dict[str, Any], operation: str) -> list[str]:
    axes = ["presence", "domain", "count", "distractor"]
    if any(area != target["area"] for area in household.get("areas") or ()):
        axes.append("area")
    if _withholdable_tool(target["capability"], operation):
        axes.append("feature")
    return [axis for axis in CONTRAST_AXES if axis in axes]


def _sole_supporter(household: dict[str, Any], target: dict[str, Any], operation: str) -> None:
    from generators.capability_registry import required_features

    dropped = set(required_features(target["capability"], operation))
    for item in household["entities"]:
        if item is not target and item["capability"] == target["capability"]:
            item["capabilities"] = [cap for cap in item["capabilities"] if cap not in dropped]


def _add_twin(household: dict[str, Any], target: dict[str, Any], noun: str, area: str) -> str | None:
    name = f"{area} {noun}"
    taken = {item["name"].casefold() for item in household["entities"]}
    taken |= {item["name"].casefold() for item in household.get("context_entities") or []}
    if name.casefold() in taken:
        return None
    twin = _clone(target)
    twin.update(
        name=name,
        aliases=[name],
        area=area,
        floor=(household.get("area_floors") or {}).get(area, target.get("floor")),
        entity_id=f"{target['domain']}.{_slug(name)}_contrast",
    )
    household["entities"].append(twin)
    return name


def _area_mutation(
    base_home: dict[str, Any], target: dict[str, Any], noun: str, rng: random.Random
) -> Any:
    others = [area for area in base_home["areas"] if area != target["area"]]
    twin_area = rng.choice(others)
    twin = _add_twin(base_home, target, noun, twin_area)
    kinds: dict[str, Any] = {"out": lambda: mutate_area(base_home, target["name"], rng.choice(others))}
    if twin:
        kinds["in"] = lambda: mutate_area(base_home, twin, target["area"])
        aside = [area for area in others if area != twin_area]
        if aside:
            kinds["aside"] = lambda: mutate_area(base_home, twin, rng.choice(aside))
    return kinds[rng.choice(sorted(kinds))]


def contrast_command(
    household: dict[str, Any], name: str, expected: dict[str, Any], noun: str, group_id: str
) -> str:
    import zlib

    from generators.utterances import finalize_training_utterance, request_seed_from_spec
    from generators.validation import check_fuzzy_eval_overlap, check_quality_eval_overlap

    spec = {
        "expected": expected,
        "target_names": [name],
        "spoken_targets": {name: f"the {noun.casefold()}"},
        "home": household,
    }
    for retry in range(_COMMAND_RETRIES):
        key = f"{group_id}:{retry}"
        seed = request_seed_from_spec({**spec, "phrasing_seed": key})
        command = finalize_training_utterance(seed, random.Random(zlib.crc32(key.encode())))
        # Same predicate the pipeline applies to synthetic rows; a rejected phrasing must not be retried as-is.
        if not (check_quality_eval_overlap(command) or check_fuzzy_eval_overlap(command)):
            return command
    raise ContrastCommandError(f"every contrast command for {group_id} overlaps an eval prompt")


def build_contrast_group(
    index: int,
    rng: random.Random,
    *,
    size: int | None = None,
    axes: tuple[str, ...] | None = None,
    variants: int | None = None,
) -> list[dict[str, Any]]:
    # Short commands such as "turn off the ceiling fan" can overlap an eval prompt under every phrasing seed,
    # so a fresh base (and household) is drawn instead; the first draw is unchanged.
    for _ in range(_GROUP_ATTEMPTS - 1):
        try:
            return _build_contrast_group(index, rng, size=size, axes=axes, variants=variants)
        except ContrastCommandError:
            continue
    return _build_contrast_group(index, rng, size=size, axes=axes, variants=variants)


def _build_contrast_group(
    index: int,
    rng: random.Random,
    *,
    size: int | None,
    axes: tuple[str, ...] | None,
    variants: int | None,
) -> list[dict[str, Any]]:
    from generators.homes import generate_home

    for _ in range(32):
        household = generate_home(index, size or rng.choice((32, 64)), rng)
        bases = _contrast_bases(household)
        if axes:
            bases = [
                base for base in bases
                if set(axes) <= set(_axes_for(household, base[0], base[2]))
            ]
        if bases:
            break
    else:
        raise ValueError(f"no contrast base for index {index}")
    by_capability: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
    for base in bases:
        by_capability.setdefault(base[0]["capability"], []).append(base)
    target, noun, operation = rng.choice(by_capability[rng.choice(sorted(by_capability))])
    name, capability = target["name"], target["capability"]
    base_home = _clone(household)
    base_home["sayso_entity_area"] = target["area"]
    base_home["context_entities"] = [
        item for item in base_home.get("context_entities") or []
        if not item["name"].casefold().startswith(name.casefold())
    ]
    available = _axes_for(base_home, target, operation)
    if axes:
        chosen = list(axes)
    elif variants:
        chosen = rng.sample(available, min(variants - 1, len(available)))
    else:
        chosen = rng.sample(available, rng.randint(1, min(4, len(available))))
    target = _find(base_home, name)
    area_mutation = _area_mutation(base_home, target, noun, rng) if "area" in chosen else None
    if "feature" in chosen:
        _sole_supporter(base_home, target, operation)
    mutations = {
        "presence": lambda: mutate_presence(base_home, name),
        "domain": lambda: mutate_domain(base_home, name, rng.choice(("sensor", "binary_sensor"))),
        "count": lambda: mutate_count(base_home, name),
        "area": area_mutation,
        "feature": lambda: mutate_feature(base_home, name, operation),
        "distractor": lambda: mutate_distractor(base_home, name),
    }
    group_id = f"contrast_{index:06d}_{_slug(name)}_{operation}"
    base_expected = contrast_gold(base_home, capability, operation, noun, seed=index)
    command = contrast_command(base_home, name, base_expected, noun, group_id)
    homes = [("base", base_home), *((axis, mutations[axis]()) for axis in chosen)]
    group = []
    for position, (axis, mutated) in enumerate(homes):
        mutated["home_id"] = f"{group_id}_{position}"
        expected = contrast_gold(mutated, capability, operation, noun, seed=index)
        group.append({
            "family": f"contrast_{axis}",
            "phrasing_seed": group_id,
            "capability": capability,
            "operation": operation,
            "targeting": "individual",
            "robustness": "ambiguity",
            "inject_missing": False,
            "utterance": command,
            "rng_index": index,
            "request_intent": {"name": noun, "area": None},
            "home": mutated,
            "target_name": name,
            "noun": noun,
            "withheld_tools": withheld_tools(mutated, capability, operation),
            "expected": expected,
            "expected_class": behaviour_class(expected),
            "contrast_group": group_id,
            "contrast_axis": axis,
            "contrast_variant": position,
        })
    return group


def seeded_contrast_group(group_index: int, seed: int) -> list[dict[str, Any]]:
    return build_contrast_group(
        group_index, random.Random(seed * 1_000_003 + group_index), variants=CONTRAST_GROUP_SIZE
    )


GENERATED_CARRIER_AXES: dict[str, tuple[str, ...]] = {
    "ordinary": ("distractor",),
    "clarify": ("count",),
    "absence": ("presence", "domain"),
}
GENERATED_INDEX_OFFSET = 1_000_000
CONTEXT_SENSOR_DOMAINS = frozenset({"sensor", "select"})


def _targets_context_sensor(item: dict[str, Any]) -> bool:
    if item["expected"].get("kind") != "action":
        return False
    sensors = {
        entity["name"] for entity in item["home"].get("context_entities") or []
        if entity.get("domain") in CONTEXT_SENSOR_DOMAINS
    }
    return item.get("target_name") in sensors


def generated_carrier_variant(family: str, index: int, seed: int) -> dict[str, Any] | None:
    from generators.gold import gold_matches_family

    # "unsupported" has no generated carrier: its blocked-request slots must keep their own
    # capability/operation, and this call does not know which slot it is filling.
    axes = GENERATED_CARRIER_AXES.get(family)
    if not axes:
        return None
    group_index = GENERATED_INDEX_OFFSET + index
    rng = random.Random(seed * 1_000_003 + group_index)
    axis = rng.choice(axes)
    try:
        item = build_contrast_group(group_index, rng, axes=(axis,))[-1]
    except ValueError:
        return None
    if not gold_matches_family(item["expected"], family) or _targets_context_sensor(item):
        return None
    return {**item, "family": f"ground_generated_{axis}"}


def grounding_capable_slot(quota: Any, rng: random.Random) -> dict[str, Any] | None:
    pairs = sorted(
        {(variant["capability"], variant["operation"]) for variant in required_training_variants()}
    )
    rng.shuffle(pairs)
    for capability, operation in pairs:
        slot = quota.take_slot(capability, operation)
        if slot is not None:
            return slot
    return None
