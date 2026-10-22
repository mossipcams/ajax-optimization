
from __future__ import annotations

import random
import re
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    SATELLITE_NAME_BRAND,
    SATELLITE_NAME_SUFFIX,
    CapabilitySpec,
)

SATELLITE_RATE = 0.30

_AREAS = (
    "Kitchen", "Living Room", "Dining Room", "Master Bedroom", "Guest Bedroom", "Kids Room",
    "Nursery", "Bathroom", "Ensuite", "Powder Room", "Office", "Study", "Den", "Family Room",
    "Garage", "Workshop", "Shed", "Basement", "Attic", "Hallway", "Landing", "Entryway", "Foyer",
    "Mudroom", "Laundry Room", "Utility Room", "Pantry", "Closet", "Porch", "Patio", "Deck",
    "Balcony", "Garden", "Backyard", "Front Yard", "Driveway", "Sunroom", "Playroom",
    "Guest Suite", "Spare Room", "Home Gym", "Music Room", "Craft Room", "Storage Room",
)
_ROLES = {
    "lights": ("Ceiling Light", "Wall Light", "Lamp", "Ceiling Lights", "Spotlights", "Pendant Light"),
    "fans": ("Ceiling Fan", "Floor Fan", "Desk Fan", "Tower Fan"),
    "switches": ("Plug", "Outlet", "Lamp Plug", "Charger Plug"),
    "covers": ("Window Blinds", "Left Window Blinds", "Right Window Blinds", "Roller Blinds"),
    "locks": ("Front Door Lock", "Back Door Lock", "Side Door Lock", "Entry Door Lock"),
    "media_players": ("TV", "Television", "OLED TV", "Streaming TV",
                      "Speaker", "Soundbar", "Smart Speaker", "Echo Dot"),
    "climate": ("Thermostat", "Heating", "Heat Pump", "Air Conditioner"),
    "vacuums": ("Robot Vacuum", "Vacuum", "Robot Cleaner"),
    "scripts": ("Good Morning", "Good Night", "Leaving Home", "Welcome Home", "Movie Time",
                "Bedtime", "All Lights Off", "Water the Plants", "Feed the Cat", "Evening Cleanup"),
    "scenes": ("Movie Night", "Dinner Time", "Reading Time", "Relax", "Night Lights", "Bedtime Lights"),
    "lawn_mowers": ("Robot Mower", "Lawn Mower", "Mower"),
    "todo_lists": ("Groceries", "Shopping List", "Household Supplies", "Weekend Chores", "Hardware Store"),
    "buttons": ("Wall Button", "Light Button", "Remote Button"),
}
_ROOM_LIGHTS = {
    "Kitchen": ("Ceiling Lights", "Counter Lights", "Under Cabinet Lights", "Island Pendants", "Sink Light"),
    "Dining Room": ("Chandelier", "Pendant Lights", "Wall Lights", "Table Lamp"),
    "Bathroom": ("Mirror Light", "Vanity Lights", "Ceiling Light", "Shower Light"),
    "Bedroom": ("Bedside Lamp", "Reading Lamp", "Ceiling Light", "Left Bedside Lamp", "Right Bedside Lamp"),
    "Living Room": ("Floor Lamp", "Table Lamp", "TV Backlight", "Ceiling Lights", "Reading Lamp"),
    "Office": ("Desk Lamp", "Monitor Backlight", "Ceiling Light", "Reading Lamp"),
    "Outside": ("Wall Light", "Wall Lights", "Floodlight", "Path Lights", "String Lights"),
}
_BRANDS = {
    "lights": ("Hue", "Ikea", "LIFX", "Nanoleaf", "Govee", "Lutron"),
    "fans": ("Dyson", "Honeywell", "Vornado"),
    "switches": ("Shelly", "Kasa", "Wyze", "Sonoff"),
    "covers": ("Ikea", "Somfy"), "locks": ("Yale", "August", "Schlage", "Aqara"),
    "media_players": ("Samsung", "LG", "Sony", "Roku", "Vizio", "Panasonic"),
    "climate": ("Nest", "Ecobee", "Honeywell", "Tado"),
    "vacuums": ("Roborock", "Roomba", "Deebot", "Eufy"),
}
_OWNERS = ("Joe's", "O'Malley's", "McKay's", "Kids'", "Children's", "Nana's", "Grandad's",
           "Priya's", "Wei's", "Amara's", "Sofia's", "Yusuf's", "Ingrid's", "Mateo's",
           "Rosa's", "Dmitri's", "Fatima's", "Kenji's", "Niamh's", "Tomas's", "Leila's",
           "Bram's", "Oona's", "Hallie's")
_OUTDOOR_AREAS = ("Garden", "Backyard", "Front Yard", "Driveway", "Walkway", "Porch", "Patio", "Deck", "Balcony", "Courtyard")
_DEVICE_AREAS = {
    "media_players": ("Living Room", "Family Room", "Den", "Lounge", "Master Bedroom", "Guest Bedroom", "Kids Room", "Media Room", "Home Theater", "Basement"),
    "climate": ("Hallway", "Living Room", "Master Bedroom", "Guest Bedroom", "Office", "Basement"),
    "vacuums": ("Living Room", "Hallway", "Utility Room", "Laundry Room", "Kitchen"),
    "locks": ("Entryway", "Foyer", "Garage", "Porch", "Mudroom"),
    "lawn_mowers": ("Garden", "Backyard", "Front Yard", "Shed", "Garage"),
    "todo_lists": ("Kitchen", "Office", "Utility Room"),
}


# Holdout-only vocabulary. Never used by default generation; holdout homes draw
# exclusively from these, which keeps them disjoint from every training pool.
_HOLDOUT_INDOOR_AREAS = (
    "Conservatory", "Snug", "Boot Room", "Loft", "Wine Cellar",
    "Gallery", "Ballroom", "Cinema Room", "Vestibule", "Potting Shed",
)
_HOLDOUT_OUTDOOR_AREAS = ("Orchard", "Terrace", "Allotment", "Greenhouse")
_HOLDOUT_AREAS = (*_HOLDOUT_INDOOR_AREAS, *_HOLDOUT_OUTDOOR_AREAS)
_HOLDOUT_DEVICE_AREAS = {
    "media_players": ("Snug", "Gallery", "Ballroom", "Cinema Room", "Loft"),
    "climate": ("Conservatory", "Loft", "Snug"),
    "vacuums": ("Boot Room", "Conservatory", "Wine Cellar"),
    "locks": ("Vestibule", "Boot Room"),
    "lawn_mowers": ("Orchard", "Allotment", "Greenhouse", "Terrace"),
    "todo_lists": ("Conservatory", "Boot Room"),
}
_HOLDOUT_ROLES = {
    "lights": ("Sconce", "Strip Light", "Uplight", "Pendant Sconce"),
    "fans": ("Fan Heater", "Pedestal Fan", "Bladeless Fan"),
    "switches": ("Smart Plug", "Extension Socket", "Mains Relay"),
    "covers": ("Roman Blind", "Shutter", "Venetian Blind"),
    "locks": ("Keypad Lock", "Smart Deadbolt", "Gate Latch"),
    "media_players": ("Projector", "Soundbar Pair", "Stereo Amplifier", "Bluetooth Speaker"),
    "climate": ("Radiator Valve", "Underfloor Heating", "Heat Recovery Unit"),
    "vacuums": ("Mop Robot", "Wet Vacuum"),
    "scripts": ("Sunrise Routine", "Guest Departure", "Shutdown Sequence", "Cellar Check"),
    "scenes": ("Cinema Mode", "Candlelight", "Brunch Setting", "Guest Arrival"),
    "lawn_mowers": ("Ride-on Mower", "Robot Scythe", "Petrol Mower"),
    "todo_lists": ("Garden Centre", "Pharmacy Run", "Dry Cleaning"),
    "buttons": ("Panic Button", "Scene Dial"),
}
_HOLDOUT_BRANDS = {
    "lights": ("Philips", "Yeelight"),
    "fans": ("Meaco",),
    "switches": ("Meross", "Wemo"),
    "covers": ("Eve",),
    "locks": ("Nuki",),
    "media_players": ("Epson", "Bose", "Denon"),
    "climate": ("Daikin", "Mitsubishi"),
    "vacuums": ("Dreame", "Shark"),
}
_HOLDOUT_OWNERS = ("Tariq's", "Ngozi's", "Saoirse's", "Ravi's", "Elif's")


def device_areas(capability: str, areas: list[str], *, holdout: bool = False) -> list[str]:
    allowed = (_HOLDOUT_DEVICE_AREAS if holdout else _DEVICE_AREAS).get(capability)
    if allowed:
        return [area for area in areas if area in allowed]
    outdoor = _HOLDOUT_OUTDOOR_AREAS if holdout else _OUTDOOR_AREAS
    if capability in {"covers", "fans"}:
        return [area for area in areas if area not in outdoor]
    return areas


def _roles(capability: str, area: str) -> tuple[str, ...]:
    if capability == "scenes":
        if area in {"Living Room", "Family Room", "Den", "Lounge", "Media Room", "Home Theater"}:
            return ("Movie Night", "Reading Lights", "Relax", "Evening Lights")
        if area in {"Kitchen", "Dining Room", "Dining Area"}:
            return ("Dinner Lights", "Bright Lights", "Evening Lights", "Night Lights")
        return ("Dimmed Lights", "Bright Lights", "Evening Lights", "Night Lights")
    if capability == "lights":
        if area in _OUTDOOR_AREAS:
            return _ROOM_LIGHTS["Outside"]
        if "Bedroom" in area or area in {"Kids Room", "Nursery"}:
            return _ROOM_LIGHTS["Bedroom"]
        if area in {"Bathroom", "Ensuite", "Powder Room", "Washroom"}:
            return _ROOM_LIGHTS["Bathroom"]
        return _ROOM_LIGHTS.get(area, _ROLES[capability])
    if capability == "fans" and area in {"Kitchen", "Bathroom", "Ensuite", "Laundry Room"}:
        return ("Extractor Fan", "Ventilation Fan", "Exhaust Fan")
    if capability == "switches" and area == "Kitchen":
        return ("Coffee Maker Plug", "Kettle Plug", "Counter Outlet", "Toaster Plug")
    return _ROLES[capability]


_ENTITY_TEMPLATES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "lights": ("Light", ("on", "off"), ("on", "off", "brightness", "color", "color_temp")),
    "fans": ("Fan", ("on", "off"), ("on", "off", "percentage")),
    "switches": ("Outlet", ("on", "off"), ("on", "off")),
    "covers": ("Blinds", ("open", "closed"), ("open", "close")),
    "locks": ("Door Lock", ("locked", "unlocked"), ("lock", "unlock")),
    "media_players": ("TV", ("off", "idle", "playing"),
                      ("on", "off", "play", "pause", "volume", "volume_step", "mute",
                       "next", "previous", "search")),
    "climate": ("Thermostat", ("heat", "cool", "off"), ("heat", "cool", "off")),
    "vacuums": ("Vacuum", ("docked", "cleaning"), ("start", "stop")),
    "scenes": ("Scene", ("off", "on"), ("activate",)),
    "scripts": ("Script", ("off", "on"), ("run",)),
    "lawn_mowers": ("Lawn Mower", ("docked", "mowing"), ("start", "stop")),
    "todo_lists": ("Shopping List", ("idle",), ("add",)),
    "buttons": ("Button", ("idle",), ("press",)),
}

_SPECIAL_NAMES: dict[str, tuple[str, ...]] = {
    "lights": ("Kids' Room Light", "O'Malley's Porch Light"),
    "fans": ("Joe's Workshop Fan", "Children's Bedroom Fan"),
    "switches": ("McKay's Office Outlet", "Joe's Desk Outlet"),
    "covers": ("Children's Bedroom Blinds", "O'Malley's Study Blinds"),
    "locks": ("McKay's Front Door Lock", "Children's Door Lock"),
    "media_players": ("Living Room TV", "Bedroom TV"),
}

_KIND_MAP: dict[str, str] = {
    "lights": "light",
    "fans": "fan",
    "switches": "switch",
    "covers": "cover",
    "locks": "lock",
    "media_players": "media_player",
    "climate": "climate",
    "vacuums": "vacuum",
    "scenes": "scene",
    "scripts": "script",
    "lawn_mowers": "lawn_mower",
    "todo_lists": "todo",
    "buttons": "button",
    "timers": "timer",
}


_NICKNAMES: dict[str, tuple[str, ...]] = {
    "ceiling lights": ("big lights", "overhead lights", "main lights"),
    "ceiling light": ("big light", "overhead light", "main light"),
    "counter lights": ("worktop lights", "cooking lights"),
    "bedside lamp": ("night lamp", "nightstand lamp"),
    "floor lamp": ("corner lamp", "standing lamp"),
    "reading lamp": ("book lamp", "chair lamp"),
    "desk lamp": ("work lamp", "study lamp"),
    "table lamp": ("side lamp",),
    "tv": ("telly", "big screen"),
    "television": ("telly", "big screen"),
    "speaker": ("music speaker",),
    "ceiling fan": ("overhead fan",),
    "desk fan": ("cooling fan",),
    "floor fan": ("cooling fan",),
    "tower fan": ("cooling fan",),
    "kettle plug": ("kettle",),
    "coffee maker plug": ("coffee machine",),
    "charger plug": ("phone charger",),
}
_NICKNAME_KEYS = sorted(_NICKNAMES, key=len, reverse=True)


def _nickname_alias(
    name: str, owners: tuple[str, ...], nickname_rng: random.Random, used: set[str]
) -> str | None:
    lowered = name.casefold()
    for key in _NICKNAME_KEYS:
        if not lowered.endswith(key):
            continue
        if nickname_rng.random() >= 0.5:
            return None
        if any(lowered.startswith(owner.casefold()) for owner in owners):
            candidates = (f"my {key}",)
        else:
            candidates = _NICKNAMES[key]
        for candidate in candidates:
            if candidate.casefold() not in used:
                return candidate
        return None
    return None


def _slug(name: str) -> str:
    return "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")


def _device_class_for(cap: CapabilitySpec) -> str | None:
    mapping = {
        "covers": "blind",
        "locks": "door",
        "switches": "outlet",
        "media_players": "tv",
    }
    return cap.device_class or mapping.get(cap.name)


_MEDIA_PROFILES: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        ("satellite",),
        "speaker",
        # No absolute "volume": a satellite is only adjusted relative to its current level.
        ("on", "off", "play", "pause", "volume_step", "mute"),
    ),
    (
        ("echo", "smart speaker", "google nest", "homepod"),
        "speaker",
        ("play", "pause", "volume", "volume_step", "mute", "next", "previous", "search"),
    ),
    (
        ("speaker", "soundbar", "sonos", "stereo", "receiver"),
        "speaker",
        ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous"),
    ),
    (
        ("streaming tv", "roku", "apple tv", "chromecast", "shield"),
        "tv",
        ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous", "search"),
    ),
)
_MEDIA_DEFAULT = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")


def media_player_profile(name: str) -> tuple[str, tuple[str, ...]]:
    lowered = name.casefold()
    for keywords, device_class, features in _MEDIA_PROFILES:
        if any(keyword in lowered for keyword in keywords):
            return device_class, features
    return "tv", _MEDIA_DEFAULT


def make_entity(
    *,
    name: str,
    capability: str,
    area: str,
    floor: str,
    rng: random.Random,
    aliases: list[str] | None = None,
    features: tuple[str, ...] | None = None,
    state: str | None = None,
    device_class: str | None = None,
) -> dict[str, Any]:
    cap = CAPABILITIES[capability]
    kind = _KIND_MAP[capability]
    noun, states, default_features = _ENTITY_TEMPLATES[capability]
    domain = cap.domain
    if capability == "media_players" and (device_class is None or features is None):
        profiled_class, profiled_features = media_player_profile(name)
        device_class = device_class or profiled_class
        features = features or profiled_features
    device_class = device_class or _device_class_for(cap)
    return {
        "entity_id": f"{domain}.{_slug(name)}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "capability": capability,
        "device_class": device_class,
        "area": area,
        "floor": floor,
        "state": state or rng.choice(states),
        "capabilities": list(features or default_features),
    }


def _eval_cases_module() -> Any:
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    try:
        from evals import cases
    except Exception:
        return None
    return cases


@lru_cache(maxsize=1)
def _eval_entity_names() -> frozenset[str]:
    # Every eval case's household, every tag, names and aliases: a synthetic name equal to any of
    # these makes the command that names it fail the generation-time eval-overlap check.
    cases = _eval_cases_module()
    if cases is None:
        return frozenset()
    names: set[str] = set()
    for household in {case.household for case in cases.load_all_cases()}:
        for entity in cases.load_home(household).get("entities") or []:
            names.add(entity["name"])
            names.update(entity.get("aliases") or [])
    return frozenset(name.casefold() for name in names)


_HOLDOUT = False


@contextmanager
def holdout_pools() -> Iterator[None]:
    global _HOLDOUT
    previous, _HOLDOUT = _HOLDOUT, True
    try:
        yield
    finally:
        _HOLDOUT = previous


def _holdout_mode(holdout: bool | None) -> bool:
    return _HOLDOUT if holdout is None else holdout


def holdout_slice() -> dict[str, frozenset[str]]:
    return {
        "areas": frozenset(_HOLDOUT_AREAS),
        "nouns": frozenset(noun for nouns in _HOLDOUT_ROLES.values() for noun in nouns),
        "owners": frozenset(_HOLDOUT_OWNERS),
        "brands": frozenset(brand for brands in _HOLDOUT_BRANDS.values() for brand in brands),
    }


_CONTEXT_PERSON_NAMES = (
    "Alice Morgan", "Ben Carter", "Chloe Nguyen", "David Osei", "Emma Lindqvist",
    "Farid Haddad", "Grace Kim", "Hugo Martins", "Ines Duarte", "Jonas Weber",
)


def _context_entities_for(
    entities: list[dict[str, Any]], rooms: list[str], index: int, size: int,
    rng_state: tuple[int, tuple[int, ...], None | Any]
) -> list[dict[str, Any]]:
    rng = random.Random(f"context:{index}:{size}:{rng_state}")
    taken: set[str] = set()
    for entity in entities:
        taken.add(entity["name"].casefold())
        for alias in entity.get("aliases", []):
            taken.add(alias.casefold())
    reserved = _eval_entity_names()
    items: list[dict[str, Any]] = []

    def add(name: str, domain: str, area: str | None = None) -> None:
        if name.casefold() in taken or name.casefold() in reserved:
            return
        taken.add(name.casefold())
        item: dict[str, Any] = {"name": name, "domain": domain}
        if area:
            item["area"] = area
        items.append(item)

    for entity in entities:
        if entity["capability"] == "media_players" and rng.random() < 0.8:
            add(f"{entity['name']} Active app", "sensor", entity.get("area"))
            add(f"{entity['name']} Application", "select", entity.get("area"))
    for entity in entities:
        if entity["capability"] in {"lights", "switches"} and rng.random() < 0.15:
            add(f"{entity['name']} {rng.choice(('Power', 'Energy'))}", "sensor", entity.get("area"))
    for area in rooms:
        if rng.random() < 0.25:
            add(f"{area} Temperature", "sensor", area)
            add(f"{area} Humidity", "sensor", area)
            add(f"{area} Motion", "binary_sensor", area)
    for name in rng.sample(_CONTEXT_PERSON_NAMES, rng.randint(1, 3)):
        add(name, "person")
    if rng.random() < 0.9:
        add("Sun", "sun")
    if rng.random() < 0.85:
        add("Date", "sensor")
    return items


def _random_entity_name(
    capability: str,
    area: str,
    slot: int,
    index: int,
    rng: random.Random,
    taken: set[str] | None = None,
    owners: tuple[str, ...] | None = None,
    bare_name_rate: float = 0.0,
    *,
    holdout: bool | None = None,
) -> str:
    holdout = _holdout_mode(holdout)
    taken = taken if taken is not None else set()
    if holdout:
        owners = owners if owners is not None else _HOLDOUT_OWNERS
        roles = _HOLDOUT_ROLES[capability]
        brands = _HOLDOUT_BRANDS.get(capability, ())
    else:
        owners = owners if owners is not None else _OWNERS
        roles = _roles(capability, area)
        brands = _BRANDS.get(capability, ())
    for attempt in range(128):
        role = rng.choice(roles)
        shape = rng.random()
        if capability == "todo_lists":
            name = role
        elif capability == "scripts":
            name = role if shape < 0.65 else f"{area} {rng.choice(('Lights Off', 'Lights On', 'Evening Lighting'))}"
        elif capability == "scenes":
            name = f"{area} {role}"
        elif shape < 0.12:
            name = f"{rng.choice(owners)} {role}"
        elif shape < 0.35 and brands:
            name = f"{area} {rng.choice(brands)} {_ENTITY_TEMPLATES[capability][0]}"
        elif rng.random() < bare_name_rate:
            name = role
        else:
            name = f"{area} {role}"
        if rng.random() < 0.26:
            name = name[:1] + name[1:].lower()
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    for number in range(2, len(taken) + 3):
        name = f"{area} {roles[0]} {number}"
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    raise ValueError("no unique fixture name")


def _capability_slots(size: int, rng: random.Random) -> list[str]:
    counts = {
        "lights": int(size * .40), "switches": int(size * .22),
        "covers": int(size * .10), "fans": int(size * .06), "media_players": int(size * .04),
        "climate": min(2, max(1, size // 24)), "locks": min(3, max(1, size // 24)),
        "scripts": rng.randint(1, 4),
        "vacuums": int(size >= 32), "buttons": int(size >= 32),
        "todo_lists": int(size >= 32), "lawn_mowers": int(size >= 64),
    }
    slots = [capability for capability, count in counts.items() for _ in range(count)]
    slots += ["lights"] * max(0, size - len(slots))
    rng.shuffle(slots)
    return slots[:size]


def _collision_aliases(name: str, area: str, noun: str) -> list[str]:
    if name.casefold().startswith(area.casefold()):
        return [f"{area} {noun}", noun]
    return [noun]


def generate_home(
    index: int,
    size: int,
    rng: random.Random,
    *,
    sayso_entity_area: str | None = None,
    bare_name_rate: float = 0.0,
    holdout: bool | None = None,
) -> dict[str, Any]:
    holdout = _holdout_mode(holdout)
    capabilities = _capability_slots(size, rng)
    entities: list[dict[str, Any]] = []
    if holdout:
        rooms = list(_HOLDOUT_AREAS)
    else:
        rooms = ["Kitchen", "Living Room", "Master Bedroom", "Bathroom", "Hallway", "Entryway", "Garage"]
        extras = [area for area in _AREAS if area not in rooms]
        rooms += rng.sample(extras, min(5, max(1, size // 8)))
    floors = {area: ("Basement" if area == "Basement" else
                     "Upstairs" if "Bedroom" in area or area == "Attic" else "Main Floor") for area in rooms}
    taken: set[str] = set()
    owners = tuple(rng.sample(_HOLDOUT_OWNERS if holdout else _OWNERS, 2))
    nickname_rng = random.Random(f"nicknames:{index}:{size}")
    used_nicknames: set[str] = set()
    for slot, capability in enumerate(capabilities):
        area = rng.choice(device_areas(capability, rooms, holdout=holdout))
        name = _random_entity_name(
            capability, area, slot, index, rng, taken, owners,
            bare_name_rate=bare_name_rate, holdout=holdout,
        )
        taken.add(_slug(name))
        noun = _ENTITY_TEMPLATES[capability][0].lower()
        bare = not name.casefold().startswith(area.casefold())
        if capability in {"scripts", "scenes", "todo_lists"} or bare:
            aliases = [name]
        else:
            aliases = list(dict.fromkeys([name, f"{area} {noun}"]))
        if capability == "lights":
            synonyms = {"light": "lamp", "lights": "lamps", "lamp": "light", "lamps": "lights",
                        "spotlights": "spots", "backlight": "backlighting"}
            alias = re.sub(r"\b(light|lights|lamp|lamps|spotlights|backlight)\b",
                           lambda match: synonyms[match[0].lower()], name, flags=re.I)
            if alias.casefold() != name.casefold():
                aliases.append(alias)
        nickname = _nickname_alias(name, owners, nickname_rng, used_nicknames)
        if nickname:
            aliases.append(nickname)
            used_nicknames.add(nickname.casefold())
        entities.append(make_entity(name=name, capability=capability, area=area,
                                    floor=floors[area], rng=rng, aliases=aliases))

    if size >= 16:
        lights = [entity for entity in entities if entity["capability"] == "lights"]
        if lights:
            base = rng.choice(lights)
            name = _random_entity_name(
                "lights", base["area"], len(entities), index, rng, taken, owners,
                bare_name_rate=bare_name_rate, holdout=holdout,
            )
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="lights",
                    area=base["area"],
                    floor=base["floor"],
                    rng=rng,
                    aliases=_collision_aliases(name, base["area"], "light"),
                )
            )
        tvs = [entity for entity in entities if entity["capability"] == "media_players"]
        if tvs:
            elsewhere = rng.choice([
                a for a in device_areas("media_players", rooms, holdout=holdout) if a != tvs[0]["area"]
            ])
            name = _random_entity_name(
                "media_players", elsewhere, len(entities), index, rng, taken, owners,
                bare_name_rate=bare_name_rate, holdout=holdout,
            )
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="media_players",
                    area=elsewhere,
                    floor=floors[elsewhere],
                    rng=rng,
                    aliases=_collision_aliases(name, elsewhere, "tv"),
                )
            )

    remove_canonical_alias_collisions(entities, avoid_eval_names=True)
    area = sayso_entity_area or entities[index % len(entities)]["area"]
    satellite_rng = random.Random(zlib.crc32(f"satellite:{index}:{size}".encode()))
    outdoor = _HOLDOUT_OUTDOOR_AREAS if holdout else _OUTDOOR_AREAS
    # Satellites are indoor speakers. The draw is taken either way so the RNG stream is unchanged.
    if satellite_rng.random() < SATELLITE_RATE and area not in outdoor:
        brand_name = f"{SATELLITE_NAME_BRAND} {SATELLITE_NAME_SUFFIX}"
        area_name = f"{area} {SATELLITE_NAME_SUFFIX}"
        candidates = [name for name in (brand_name, area_name)
                      if _slug(name) not in taken and name.casefold() not in _eval_entity_names()]
        sat_name = satellite_rng.choice(candidates) if candidates else None
    else:
        sat_name = None
    if sat_name is not None:
        taken.add(_slug(sat_name))
        entities.append(
            make_entity(
                name=sat_name,
                capability="media_players",
                area=area,
                floor=floors[area],
                rng=satellite_rng,
                aliases=[sat_name],
            )
        )
    return {
        "home_id": f"home_{index:06d}_{size}",
        "size": size,
        "sayso_entity_area": area,
        "entities": entities,
        "active_timers": _synthetic_timers(rng, holdout) if rng.random() < 0.3 else [],
        "areas": rooms, "area_floors": floors, "owners": owners,
        "exposure_source": "synthetic",
        "context_entities": _context_entities_for(entities, rooms, index, size, rng.getstate()),
    }



def remove_canonical_alias_collisions(
    entities: list[dict[str, Any]], *, avoid_eval_names: bool = False
) -> None:
    # Eval-name avoidance is for synthetic generation only. Real homes reach this function too, and
    # their fixture names (e.g. 'TV') are eval names that must survive unchanged.
    canonical = {entity["name"].casefold() for entity in entities}
    eval_names = _eval_entity_names() if avoid_eval_names else frozenset()
    for entity in entities:
        other_names = canonical - {entity["name"].casefold()}
        entity["aliases"] = [alias for alias in entity.get("aliases", [])
                             if alias.casefold() not in other_names
                             and alias.casefold() not in eval_names]

def _synthetic_timers(rng: random.Random, holdout: bool = False) -> list[dict[str, Any]]:
    names = ("Pizza Timer", "Laundry Timer", "Kids Bath Timer")
    return [
        {
            "name": rng.choice(names),
            "remaining_seconds": rng.randint(60, 1800),
            "area": rng.choice(_HOLDOUT_AREAS if holdout else _AREAS),
        }
        for _ in range(rng.randint(1, 2))
    ]


def entities_of_capability(home: dict[str, Any], capability: str) -> list[dict[str, Any]]:
    return [e for e in home["entities"] if e.get("capability") == capability]


def entities_in_area(home: dict[str, Any], capability: str, area: str) -> list[dict[str, Any]]:
    return [e for e in entities_of_capability(home, capability) if e["area"] == area]
