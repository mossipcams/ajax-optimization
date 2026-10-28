
from __future__ import annotations

import re
import random
import zlib
from typing import Any

from generators.ohf import render_call
from generators.capability_registry import is_satellite

_CONVERSATIONAL = (
    "Hey, could you {action}.",
    "When you get a chance, {action}.",
    "Please {action}.",
    "Can you {action} for me?",
)


def vary_training_utterance(text: str, rng: random.Random) -> str:
    text = text[:1].lower() + text[1:]
    if re.match(r"^(how|is|are|does|do|did|which)\b", text) or text.endswith("?"):
        return text
    for technical, spoken in (("climates", "thermostats"), ("switchs", "outlets"), ("covers", "blinds"), ("media players", "TVs")):
        text = text.replace(f"the {technical} ", f"the {spoken} ")
    if " and " not in text and ", but leave " not in text:
        patterns = (
            (r"turn on (.+)", ("turn on {0}", "turn {0} on", "switch {0} on", "switch on {0}")),
            (r"turn off (.+)", ("turn off {0}", "turn {0} off", "switch {0} off", "switch off {0}")),
            (r"set (.+) brightness to (.+) percent", ("set {0} brightness to {1} percent", "dim {0} to {1} percent", "set {0} to {1} percent brightness")),
            (r"set (.+) color to (.+)", ("set {0} color to {1}", "make {0} {1}", "change {0} to {1}")),
            (r"set (.+) color temperature to (\d+)(?: kelvin)?", ("set {0} color temperature to {1} kelvin", "set {0} to {1} kelvin")),
            (r"set (.+) temperature to (.+) degrees", ("set {0} temperature to {1} degrees", "set {0} to {1} degrees", "adjust {0} to {1} degrees")),
            (r"what is the status of (.+)", ("what is the status of {0}", "what's the status of {0}", "check the status of {0}", "tell me the status of {0}")),
        )
        for pattern, alternatives in patterns:
            match = re.fullmatch(pattern, text)
            if match:
                if "brightness" in pattern and match[2].isdigit() and int(match[2]) > 50:
                    alternatives = tuple(choice for choice in alternatives if not choice.startswith("dim "))
                text = rng.choice(alternatives).format(*match.groups())
                break
    if text.startswith("pause ") and "timer" in text and rng.random() < 0.4:
        text = rng.choice(("hold ", "freeze ")) + text[len("pause "):]
    if text.startswith("run ") and not re.search(r"\b(?:on|off)\b", text[4:].casefold()) and rng.random() < 0.2:
        text = ("turn on " if rng.random() < 0.5 else "switch on ") + text[4:]
    variants = {
        "turn on ": ("turn on ", "switch on "),
        "turn off ": ("turn off ", "switch off "),
        "run ": ("run ", "start ", "activate "),
        "play ": ("play ", "resume "),
        "what is the status of ": ("what is the status of ", "what is the current state of "),
    }
    for prefix, alternatives in variants.items():
        if text.startswith(prefix):
            text = rng.choice(alternatives) + text[len(prefix):]
            break
    template = rng.choice(("{action}", "{action}", "{action}", "please {action}", "could you {action}?", "can you {action} for me?"))
    if template != "{action}" and text.startswith(("what ", "what's ")):
        text = re.sub(r"^what(?: is|'s) ", "tell me ", text)
    return template.format(action=text)


def finalize_training_utterance(text: str, rng: random.Random, *, upper: bool | None = None) -> str:
    utterance = vary_training_utterance(text, rng)
    drawn = rng.random() >= 0.5
    return utterance[:1].upper() + utterance[1:] if (drawn if upper is None else upper) else utterance.lower()


def apply_bare_media_room_phrasing(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("spoken_targets") or spec.get("category") == "ambiguity":
        return spec
    expected = spec.get("expected") or {}
    if expected.get("kind") != "action":
        return spec
    by_name = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    seed = str(spec.get("candidate_id") or spec.get("semantic_id") or spec.get("seed", 0))
    rng = random.Random(zlib.crc32(seed.encode()))
    spoken = dict(spec.get("spoken_targets") or {})
    for name in spec.get("target_names") or []:
        entity = by_name.get(name)
        if entity is None or entity.get("capability") != "media_players":
            continue
        area = entity.get("area") or ""
        if not area or name.casefold().startswith(area.casefold()):
            continue
        if rng.random() >= 0.55:
            continue
        noun = (entity.get("device_class") or "tv").casefold()
        spoken[name] = f"the {area.casefold()} {noun}"
    if spoken != (spec.get("spoken_targets") or {}):
        spec["spoken_targets"] = spoken
    return spec


def apply_generic_wording(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.homes import _ENTITY_TEMPLATES

    capability = spec.get("capability")
    if capability in (None, "timers") or not spec.get("target_names"):
        return spec
    area = (spec.get("request_intent") or {}).get("area") or spec["home"]["sayso_entity_area"]
    by_name = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    default_noun = _ENTITY_TEMPLATES[capability][0].lower()
    spoken = {}
    for name in spec["target_names"]:
        device_class = (by_name.get(name) or {}).get("device_class")
        noun = (spec.get("request_intent") or {}).get("name") or (
            device_class if capability == "media_players" and device_class else default_noun
        )
        spoken[name] = f"the {area} {noun}"
    spec["spoken_targets"] = spoken
    return spec


def request_seed_from_spec(spec: dict[str, Any]) -> str:
    provenance: list[dict[str, Any]] = []
    spec["linguistics"] = provenance
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        return spec.get("request_hint") or _no_action_hint(expected)
    targets = [
        spec.get("spoken_targets", {}).get(name, name)
        for name in spec.get("target_names", [])
    ]
    if expected.get("kind") == "status":
        provenance.append({"source": "sayso_fallback", "intent": "GetLiveContext"})
        if targets:
            domain = ((expected.get("calls") or [{}])[0].get("arguments") or {}).get("domain")
            rng = random.Random(zlib.crc32(f"status:{spec.get('candidate_id', '')}".encode()))
            questions = _STATUS_QUESTIONS.get(domain if isinstance(domain, str) else "", ())
            if questions and rng.random() < 0.5:
                question = rng.choice(questions).format(targets[0])
                suffix = rng.choice(("", "", " right now")) if question.startswith("is ") else ""
                return question + suffix + "?"
            return f"what is the status of {targets[0]}"
        call = (expected.get("calls") or [{}])[0]
        domain = (call.get("arguments") or {}).get("domain")
        if isinstance(domain, list) and domain:
            return f"what is the status of the {domain[0]}"
        return "what is the device status"
    phrases: list[str] = []
    scope = spec.get("exclusion_scope")
    if scope and spec.get("excluded_names"):
        seed_key = spec.get("phrasing_seed") or spec.get("candidate_id", "")
        first = (expected.get("calls") or [{}])[0]
        return _phrase_for_call(scope, first, f"{seed_key}:0", provenance) + _leave_clause(spec["excluded_names"], spec)
    satellites = _satellite_names(spec)
    for index, call in enumerate(expected.get("calls") or []):
        canonical = (call.get("arguments") or {}).get("name")
        seed_key = spec.get("phrasing_seed") or spec.get("candidate_id", "")
        if canonical in satellites and call["name"] == "HassSetVolumeRelative":
            if provenance is not None:
                provenance.append({"source": "sayso_fallback", "intent": call["name"]})
            phrases.append(_self_volume_phrasing(call, f"{seed_key}:{index}"))
            continue
        target = targets[index] if index < len(targets) else ""
        if canonical in spec.get("spoken_targets", {}):
            target = spec["spoken_targets"][canonical]
        elif canonical in spec.get("target_names", []):
            target = canonical
        room = _timer_room_phrasing(spec, call, f"{seed_key}:{index}")
        if room is not None:
            if provenance is not None:
                provenance.append({"source": "sayso_fallback", "intent": call["name"]})
            phrases.append(room)
            continue
        routine = _routine_phrasing(spec, call, f"{seed_key}:{index}")
        if routine is not None:
            if provenance is not None:
                provenance.append({"source": "sayso_fallback", "intent": call["name"]})
            phrases.append(routine)
            continue
        phrases.append(_phrase_for_call(str(target), _lock_phrasing(call, spec), f"{seed_key}:{index}", provenance))
    seed = " and ".join(phrases)
    excluded = spec.get("excluded_names") or []
    if excluded:
        seed += _satellite_contrast_clause(spec, excluded) or _leave_clause(excluded, spec)
    return seed


_LEAVE_CLAUSES: tuple[str, ...] = (
    ", but leave {n} alone",
    ", but leave {n} unchanged",
    ", but leave {n} as it is",
    ", but leave {n} how it is",
    ", but not {n}, leave {p}",
)


def _leave_clause(excluded: list[str], spec: dict[str, Any]) -> str:
    seed_key = spec.get("phrasing_seed") or spec.get("candidate_id", "")
    rng = random.Random(zlib.crc32(f"leave:{seed_key}".encode()))
    return rng.choice(_LEAVE_CLAUSES).format(n=" and ".join(excluded), p="it" if len(excluded) == 1 else "them")


def _lock_phrasing(call: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments") or {}
    if call.get("name") not in {"HassTurnOn", "HassTurnOff"}:
        return call
    by_name = {entity["name"]: entity for entity in (spec.get("home") or {}).get("entities", [])}
    if arguments.get("domain") != ["lock"] and (by_name.get(arguments.get("name")) or {}).get("domain") != "lock":
        return call
    return {**call, "arguments": {**arguments, "domain": ["lock"], "device_class": ["door"]}}


def _no_action_hint(expected: dict[str, Any]) -> str:
    response = expected.get("response", "unsupported")
    hints = {
        "unsupported": "play music in the garage",
        "refuse": "disable the smoke alarm safety system",
        "clarify": "turn on the light",
        "area_unavailable": "turn on the light",
    }
    return hints.get(response, "do something unsupported")


_STATUS_QUESTIONS: dict[str, tuple[str, ...]] = {
    "lock": ("is {0} locked", "is {0} unlocked", "is {0} still locked"),
    "light": ("is {0} on", "is {0} off", "is {0} still on", "are {0} still on"),
    "switch": ("is {0} on", "is {0} off", "is {0} still on"),
    "fan": ("is {0} on", "is {0} running", "is {0} still on"),
    "cover": ("is {0} open", "is {0} closed", "are {0} open"),
    "media_player": ("is {0} playing", "is {0} on", "is anything playing on {0}"),
    "climate": ("what is {0} set to", "is {0} heating", "is {0} on"),
    "vacuum": ("is {0} cleaning", "is {0} docked", "is {0} running"),
    "lawn_mower": ("is {0} mowing", "is {0} docked"),
    "todo": ("what's on {0}", "is anything on {0}"),
}

_VACUUM_START = (
    "tell {0} to start cleaning", "start {0} cleaning", "start cleaning with {0}",
    "send {0} out to clean", "tell {0} to clean",
)

_VACUUM_RETURN_TO_BASE = (
    "send {0} back to its dock", "tell {0} to dock", "send {0} back to its base", "tell {0} to go back and charge",
)

_VACUUM_CLEAN_AREA = (
    "clean the {area} with {0}", "tell {0} to vacuum the {area}", "tell {0} to clean the {area}",
)

_ROUTINE_ACTIONS = ("run my {0} routine", "start the {0} routine", "run the routine called {0}", "go ahead and kick off {0}", "turn on my {0} routine")

_FRACTION_WORDS: dict[int, str] = {25: "a quarter", 33: "a third", 50: "half"}

_LEVEL_PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "HassLightSet": {
        "word": ("set {0} brightness to {1}", "dim {0} to {1}"),
        "down": ("dim {0} down to {1} percent", "bring {0} down to {1} percent"),
        "up": ("bring {0} up to {1} percent", "raise {0} up to {1} percent"),
    },
    "HassFanSetSpeed": {
        "word": ("set {0} speed to {1}", "set {0} to {1} speed"),
        "down": ("set {0} speed down to {1} percent", "turn {0} down to {1} percent speed"),
        "up": ("set {0} speed up to {1} percent", "turn {0} up to {1} percent speed"),
    },
}

_TV_SHOW_PHRASES: dict[str, tuple[str, ...]] = {
    "HassMediaPause": ("stop the show on {target}", "stop the show playing on {target}"),
    "HassMediaUnpause": ("continue the show on {target}", "unpause the show on {target}"),
}

_UNNAMED_TIMER_PHRASES: dict[str, tuple[str, ...]] = {
    "HassStartTimer": ("start a timer for {delta}", "set a {delta} timer"),
    "HassPauseTimer": ("hold the timer", "put my timer on pause"),
    "HassUnpauseTimer": ("resume my timer", "continue the timer"),
    "HassCancelTimer": ("cancel my timer", "kill my timer"),
    "HassTimerStatus": ("how much time is left", "check on my timer"),
    "HassIncreaseTimer": ("add {delta} to the timer", "give me {delta} more"),
    "HassDecreaseTimer": ("take {delta} off the timer", "cut {delta} off my timer"),
}

_SPOKEN_PLURALS = {"switch": "outlets", "climate": "thermostats", "cover": "blinds",
                   "media_player": "TVs", "device_class": "devices"}


def _plural(noun: str) -> str:
    return _SPOKEN_PLURALS.get(noun, noun.replace("_", " ") + "s")


def _duration(arguments: dict[str, Any]) -> str:
    return " and ".join(
        f"{arguments[unit]} {unit[:-1] if arguments[unit] == 1 else unit}"
        for unit in ("hours", "minutes", "seconds") if arguments.get(unit)
    )


def _timer_room_areas(home: dict[str, Any]) -> list[str]:
    from generators.context import OVERVIEW_EXCLUDED_DOMAINS

    areas: list[str] = []
    for source in ((home or {}).get("entities") or [], (home or {}).get("context_entities") or []):
        for entity in source:
            if entity.get("domain") in OVERVIEW_EXCLUDED_DOMAINS:
                continue
            area = entity.get("area")
            if area and area != "Unassigned" and area not in areas:
                areas.append(area)
    return areas


_TIMER_ROOM_PHRASES = (
    "set a {delta} timer in the {area}",
    "start a {delta} timer in the {area}",
)


def _timer_room_phrasing(spec: dict[str, Any], call: dict[str, Any], seed: str) -> str | None:
    arguments = call.get("arguments") or {}
    if call["name"] != "HassStartTimer":
        return None
    units = [unit for unit in ("hours", "minutes", "seconds") if arguments.get(unit)]
    if len(units) != 1 or set(arguments) - {"hours", "minutes", "seconds"}:
        return None
    areas = _timer_room_areas(spec.get("home") or {})
    if not areas:
        return None
    rng = random.Random(zlib.crc32(f"timer-room:{seed}".encode()))
    if rng.random() >= 0.25:
        return None
    unit = units[0]
    delta = f"{arguments[unit]} {unit[:-1]}"
    return rng.choice(_TIMER_ROOM_PHRASES).format(delta=delta, area=rng.choice(areas))

_VOLUME_STEP_PHRASES: dict[str, tuple[str, ...]] = {
    "up": ("turn {target} volume up {n} percent", "make {target} louder by {n}",
           "raise {target} volume by {n} percent"),
    "down": ("turn {target} volume down {n} percent", "lower {target} volume by {n}",
             "{target}, turn it down {n}"),
}

_VOLUME_UP_PHRASES = ("turn up {0} volume", "make {0} a lot louder")

_VOLUME_DOWN_PHRASES = ("turn down {0} volume", "not so loud, {0}")


def _volume_phrasing(target: str, call: dict[str, Any], seed: str) -> str:
    step = (call.get("arguments") or {}).get("volume_step")
    rng = random.Random(zlib.crc32(f"volume:{seed}".encode()))
    if isinstance(step, int):
        return rng.choice(_VOLUME_STEP_PHRASES["up" if step > 0 else "down"]).format(target=target, n=abs(step))
    if step == "up":
        return rng.choice(_VOLUME_UP_PHRASES).format(target)
    if step == "down":
        return rng.choice(_VOLUME_DOWN_PHRASES).format(target)
    return f"adjust {target} volume"


# Satellites are only adjusted relative to their current level, so self-volume is up/down only.
# Eval phrasings ("speak louder", "you're too loud") are deliberately left out.
_SELF_VOLUME_UP = ("turn yourself up", "turn yourself up a bit", "speak up", "speak up a bit", "make yourself a little louder")
_SELF_VOLUME_DOWN = ("turn yourself down", "turn yourself down a notch", "turn yourself down a bit", "you're a bit too loud", "quiet down a little")


def _self_volume_phrasing(call: dict[str, Any], seed: str) -> str:
    arguments = call.get("arguments") or {}
    rng = random.Random(zlib.crc32(f"self-volume:{seed}".encode()))
    step = arguments.get("volume_step")
    if isinstance(step, int):
        return f"turn yourself {'up' if step > 0 else 'down'} {abs(step)} percent"
    return rng.choice(_SELF_VOLUME_UP if step != "down" else _SELF_VOLUME_DOWN)


def _satellite_names(spec: dict[str, Any]) -> set[str]:
    return {entity["name"] for entity in (spec.get("home") or {}).get("entities", []) if is_satellite(entity)}


def _satellite_contrast_clause(spec: dict[str, Any], excluded: list[str]) -> str | None:
    calls = (spec.get("expected") or {}).get("calls") or []
    if len(calls) != 1 or len(excluded) != 1:
        return None
    target = (calls[0].get("arguments") or {}).get("name")
    satellites = _satellite_names(spec)
    if target in satellites and excluded[0] not in satellites:
        return f", not the {excluded[0]}"
    if excluded[0] in satellites and target and target not in satellites:
        return ", not you"
    return None


def _vacuum_paraphrase(target: str, call: dict[str, Any], seed: str) -> str | None:
    name, arguments = call["name"], call.get("arguments") or {}
    if not target:
        return None
    rng = random.Random(zlib.crc32(f"vacuum-paraphrase:{seed}".encode()))
    if name == "HassVacuumReturnToBase" and rng.random() < 0.6:
        return rng.choice(_VACUUM_RETURN_TO_BASE).format(target)
    if name == "HassVacuumCleanArea" and arguments.get("area") and rng.random() < 0.6:
        return rng.choice(_VACUUM_CLEAN_AREA).format(target, area=arguments["area"])
    return None


def _level_phrasing(target: str, call: dict[str, Any], seed: str) -> str | None:
    name, arguments = call["name"], call.get("arguments") or {}
    if name not in _LEVEL_PHRASES or not target:
        return None
    key = "brightness" if name == "HassLightSet" else "percentage"
    level = arguments.get(key)
    if not isinstance(level, int) or set(arguments) - {"name", "area", "floor", "domain", "device_class", key}:
        return None
    rng = random.Random(zlib.crc32(f"level:{seed}".encode()))
    if rng.random() >= 0.5:
        return None
    word = _FRACTION_WORDS.get(level)
    if word:
        return rng.choice(_LEVEL_PHRASES[name]["word"]).format(target, word)
    return rng.choice(_LEVEL_PHRASES[name]["down" if level <= 50 else "up"]).format(target, level)


def _routine_phrasing(spec: dict[str, Any], call: dict[str, Any], seed: str) -> str | None:
    from generators.tools import script_tool_name

    entities = (spec.get("home") or {}).get("entities", [])
    matches = [entity for entity in entities if entity.get("domain") == "script" and script_tool_name(entity) == call["name"]]
    if len(matches) != 1:
        return None
    name = matches[0]["name"]
    folded = name.casefold()
    if "routine" in folded or sum(entity["name"].casefold() == folded for entity in entities) != 1:
        return None
    if any(folded in {alias.casefold() for alias in entity.get("aliases") or []} for entity in entities if entity is not matches[0]):
        return None
    rng = random.Random(zlib.crc32(f"routine:{seed}".encode()))
    if rng.random() >= 0.5:
        return None
    actions = _ROUTINE_ACTIONS
    if re.search(r"\b(?:on|off)\b", folded):
        actions = tuple(action for action in _ROUTINE_ACTIONS if not action.startswith("turn on"))
    return rng.choice(actions).format(name)


def _phrase_for_call(target: str, call: dict[str, Any], seed: str = "", provenance=None) -> str:
    if call["name"] == "HassVacuumStart" and target:
        rng = random.Random(zlib.crc32(f"vacuum:{seed}".encode()))
        if rng.random() < 0.6:
            if provenance is not None:
                provenance.append({"source": "sayso_fallback", "intent": call["name"]})
            return rng.choice(_VACUUM_START).format(target)
    if call["name"] in _TV_SHOW_PHRASES and target and "tv" in (call.get("arguments") or {}).get("device_class", []):
        rng = random.Random(zlib.crc32(f"tv_show:{seed}".encode()))
        if rng.random() < 0.3:
            if provenance is not None:
                provenance.append({"source": "sayso_fallback", "intent": call["name"]})
            return rng.choice(_TV_SHOW_PHRASES[call["name"]]).format(target=target)
    phrasing = _vacuum_paraphrase(target, call, seed) or _level_phrasing(target, call, seed)
    if phrasing is not None:
        if provenance is not None:
            provenance.append({"source": "sayso_fallback", "intent": call["name"]})
        return phrasing
    rendered = render_call(call, target, seed, provenance=provenance)
    if rendered is not None:
        if rendered.lower().startswith("next track"):
            return "skip to the " + rendered
        if rendered.lower().startswith("previous track"):
            return "go back to the " + rendered
        return rendered
    if provenance is not None:
        provenance.append({"source": "sayso_fallback", "intent": call["name"]})
    name, arguments = call["name"], call.get("arguments") or {}
    if name in _UNNAMED_TIMER_PHRASES and not (arguments.get("name") or arguments.get("area") or arguments.get("floor")):
        rng = random.Random(zlib.crc32(f"timer:{seed}".encode()))
        return rng.choice(_UNNAMED_TIMER_PHRASES[name]).format(delta=_duration(arguments))
    if not target and (arguments.get("area") or arguments.get("floor")):
        domain = arguments.get("domain") or arguments.get("device_class") or ["device"]
        noun = domain[0] if isinstance(domain, list) else domain
        target = f"the {_plural(noun)}"
    if arguments.get("area"):
        target += f" in {arguments['area']}"
    if arguments.get("floor"):
        target += f" on {arguments['floor']}"
    if not name.startswith(("Hass", "Get")):
        return f"run {target}"
    device_class = set(arguments.get("device_class") or [])
    if name == "HassTurnOn":
        if "door" in device_class:
            return f"lock {target}"
        if device_class & {"blind", "garage", "curtain", "shade"}:
            return f"open {target}"
        return f"turn on {target}"
    if name == "HassTurnOff":
        if "door" in device_class:
            return f"unlock {target}"
        if "garage" in device_class:
            return f"close {target}"
        if device_class:
            return f"close {target}"
        return f"turn off {target}"
    if name == "HassLightSet":
        settings = []
        if "brightness" in arguments:
            settings.append(f"brightness to {arguments['brightness']} percent")
        if "color" in arguments:
            settings.append(f"color to {arguments['color']}")
        if "temperature" in arguments:
            settings.append(f"color temperature to {arguments['temperature']} kelvin")
        if settings:
            return f"set {target} " + " and ".join(settings)
    if name == "HassFanSetSpeed":
        return f"set {target} speed to {arguments['percentage']} percent"
    if name == "HassClimateSetTemperature":
        return f"set {target} temperature to {arguments['temperature']} degrees"
    if name == "HassMediaPause":
        return f"pause {target}"
    if name == "HassMediaUnpause":
        return f"play {target}"
    if name == "HassSetVolume":
        return f"set {target} volume to {arguments['volume_level']} percent"
    if name == "HassSetVolumeRelative":
        return _volume_phrasing(target, call, seed)
    if name == "HassMediaPlayerMute":
        return f"mute {target}"
    if name == "HassMediaPlayerUnmute":
        return f"unmute {target}"
    if name == "HassMediaNext":
        return f"skip to the next track on {target}"
    if name == "HassMediaPrevious":
        return f"go back to the previous track on {target}"
    if name == "HassMediaSearchAndPlay":
        return f"search for {arguments['search_query']} on {target}"
    if name == "HassStartTimer":
        duration = _duration(arguments)
        suffix = f" called {arguments['name']}" if arguments.get("name") else ""
        suffix += "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        if duration:
            return f"start a timer for {duration}{suffix}"
        return f"start a timer{suffix}"
    if name == "HassPauseTimer":
        scope = "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        return f"pause the {arguments.get('name', '')} timer{scope}".replace("  ", " ")
    if name == "HassUnpauseTimer":
        scope = "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        return f"resume the {arguments.get('name', '')} timer{scope}".replace("  ", " ")
    if name == "HassCancelTimer":
        scope = "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        return f"cancel the {arguments.get('name', '')} timer{scope}".replace("  ", " ")
    if name in {"HassIncreaseTimer", "HassDecreaseTimer"}:
        delta = _duration(arguments)
        verb = "add" if name == "HassIncreaseTimer" else "take"
        preposition = "to" if name == "HassIncreaseTimer" else "off"
        return f"{verb} {delta} {preposition} the {arguments.get('name', '')} timer".replace("  ", " ")
    if name == "HassTimerStatus":
        scope = "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        return f"what is the {arguments.get('name', '')} timer status{scope}".replace("  ", " ")
    if name == "HassVacuumStart":
        return f"start {target}"
    if name == "HassVacuumReturnToBase":
        return f"send {target} home"
    if name == "HassVacuumCleanArea":
        area = arguments.get("area")
        return f"vacuum {area}" if area else f"vacuum with {target}"
    if name == "HassCancelAllTimers":
        area = arguments.get("area")
        return f"cancel all timers in {area}" if area else "cancel all timers"
    if name == "GetLiveContext":
        return f"what is the status of {target}"
    area = arguments.get("area")
    if area and not target:
        domain = arguments.get("domain") or ["device"]
        domain_label = domain[0] if isinstance(domain, list) else domain
        floor = arguments.get("floor")
        if name == "HassTurnOn":
            op = "turn on"
        elif name == "HassTurnOff":
            op = "turn off"
        elif name == "HassLightSet":
            op = "set"
        else:
            op = "control"
        if floor:
            return f"{op} the {_plural(domain_label)} on {floor} in {area}"
        return f"{op} the {_plural(domain_label)} in {area}"
    return f"control {target}"


def expand_utterance(spec: dict[str, Any]) -> str:
    spec = apply_bare_media_room_phrasing(spec)
    category = spec.get("category", "clean_direct")
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        hint = spec.get("request_hint") or _no_action_hint(expected)
        return hint
    if category == "ambiguity" and spec.get("request_hint"):
        return spec["request_hint"]
    if category == "conversational":
        action = request_seed_from_spec(spec)
        index = zlib.crc32(str(spec.get("candidate_id", "")).encode()) % len(_CONVERSATIONAL)
        template = _CONVERSATIONAL[index]
        return template.format(action=action)
    seed = request_seed_from_spec(spec)
    if category == "clean_direct" and seed:
        return seed[0].upper() + seed[1:]
    return seed


_DESCRIPTIONS: tuple[str, ...] = (
    "the {mod} {domain} in the {place}",
    "the {mod} {domain} in my {place}",
    "the {place} {mod} {domain}",
    "the {brand} {domain} in the {place}",
    "the {brand} {domain} in my {place}",
    "the {place} {brand} {domain}",
)

_BRAND_TOKENS: frozenset[str] = frozenset(
    {"nanoleaf", "lutron", "ikea", "govee", "lifx", "hue", "sengled", "wiz"}
)

_STOP_TOKENS: frozenset[str] = frozenset(
    {"light", "lights", "lamp", "lamps", "lighting", "the", "a", "an", "s",
     "tv", "television", "outlet", "outlets", "plug", "plugs", "blinds",
     "blind", "shade", "shades", "curtain", "curtains", "thermostat",
     "thermostats", "lock", "locks", "fan", "fans", "vacuum", "scene",
     "script", "button", "and", "of", "my", "in", "on", "for"}
)


_DISCRIMINATION_VERBS: tuple[str, ...] = (
    "turn on", "turn off", "open", "close", "lock", "unlock", "set", "run",
)
_OFF_DISCRIMINATION_VERBS: frozenset[str] = frozenset({"turn off", "close", "unlock"})


def discrimination_utterance(call_name: str, seed: str, description: str) -> str:
    off = call_name.rsplit("__", 1)[-1] == "HassTurnOff"
    verb = "turn off" if off else "turn on"
    lowered = seed.lower()
    for candidate in _DISCRIMINATION_VERBS:
        if lowered.startswith(candidate):
            if candidate in {"set", "run"} or (candidate in _OFF_DISCRIMINATION_VERBS) == off:
                verb = candidate
            break
    return f"{verb} {description}"


def describe_target(
    domain: str,
    place: str,
    rng: random.Random,
    modifier: str | None = None,
    brand: str | None = None,
) -> str:
    noun = _plural(domain) if domain else "devices"
    order = list(range(len(_DESCRIPTIONS)))
    rng.shuffle(order)
    for index in order:
        template = _DESCRIPTIONS[index]
        needs_mod = "{mod}" in template
        needs_brand = "{brand}" in template
        if needs_mod and not modifier:
            continue
        if needs_brand and not brand:
            continue
        return template.format(
            domain=noun,
            place=place or "room",
            mod=modifier or "",
            brand=brand or "",
        )
    return ""

