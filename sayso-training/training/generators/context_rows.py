from __future__ import annotations

from collections import Counter
from itertools import combinations

import random
from typing import Any

from generators.capability_registry import CAPABILITIES
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker, pair_hash
from generators.homes import generate_home
from generators.labels import render_example
from generators.rendering import ha_brightness_pct
from generators.tools import build_turn_off, build_turn_on
from generators.utterances import finalize_training_utterance
from generators.validation import (
    check_quality_eval_overlap,
    validate_row,
    validate_serialized_row,
    validate_spec,
    validate_utterance,
)

def casing_kind(has_calls: bool) -> str:
    return "call" if has_calls else "no_call"


def _balanced_upper(casing_counts: dict[str, Counter[str]] | None, spec: dict[str, Any]) -> bool | None:
    if casing_counts is None:
        return None
    counts = casing_counts[casing_kind(bool((spec.get("expected") or {}).get("calls")))]
    if counts["upper"] - counts["lower"] >= 2:
        return False
    if counts["lower"] - counts["upper"] >= 2:
        return True
    return None


_TIMER_CONTEXT_NAMES = ("tea", "rice", "pasta", "bread", "coffee", "eggs", "pizza", "laundry")
_TIMER_CONTEXT_MINUTES = (5, 10, 15, 20, 30)
_TIMER_CONTEXT_START_NAMED = ("set a {minutes} minute {name} timer", "start a {name} timer for {minutes} minutes")
_TIMER_CONTEXT_START_UNNAMED = ("set a {minutes} minute timer", "start a timer for {minutes} minutes")
_TIMER_CONTEXT_ADJUST_HISTORY = ("add {minutes} minutes", "give it {minutes} more minutes")
_TIMER_CONTEXT_TOOLS = {
    "pause": "HassPauseTimer",
    "unpause": "HassUnpauseTimer",
    "cancel": "HassCancelTimer",
    "increase": "HassIncreaseTimer",
    "decrease": "HassDecreaseTimer",
}
_TIMER_CONTEXT_BARE = {
    "pause": ("pause it", "pause that", "pause that timer"),
    "unpause": ("resume it", "unpause it"),
    "cancel": ("cancel it", "cancel that"),
    "increase": ("add {minutes} minutes", "give it {minutes} more minutes"),
    "decrease": ("take {minutes} minutes off", "take {minutes} off it"),
}
_TIMER_CONTEXT_NAMED = {
    "pause": ("pause the {name} timer", "pause my {name} timer"),
    "cancel": ("cancel the {name} timer", "cancel my {name} timer"),
    "increase": ("add {minutes} minutes to the {name} timer", "give the {name} timer {minutes} more minutes"),
    "decrease": ("take {minutes} minutes off the {name} timer",),
}
_TIMER_CONTEXT_RAISE = ("no, make it 15 minutes, five more", "no, make it 15, that's five more")
_TIMER_CONTEXT_LOWER = ("no, add five not ten", "no, add five minutes instead of ten")


def _casing_only_text(text: str, rng: random.Random, *, upper: bool | None) -> str:
    """Casing only: finalize_training_utterance wraps text in "could you"/"please", which garbles "no, ..." corrections and "the other one too"."""
    drawn = rng.random() >= 0.5 if upper is None else upper
    return text[:1].upper() + text[1:] if drawn else text.lower()


def _timer_context_start(rng: random.Random, name: str | None, minutes: int) -> dict[str, Any]:
    if name is None:
        utterance = rng.choice(_TIMER_CONTEXT_START_UNNAMED).format(minutes=minutes)
        arguments: dict[str, Any] = {"minutes": minutes}
    else:
        utterance = rng.choice(_TIMER_CONTEXT_START_NAMED).format(name=name, minutes=minutes)
        arguments = {"name": name, "minutes": minutes}
    return {
        "utterance": utterance,
        "operation": "start",
        "call": {"name": "HassStartTimer", "arguments": arguments},
    }


def _timer_context_adjust_history(rng: random.Random, operation: str, minutes: int) -> dict[str, Any]:
    return {
        "utterance": rng.choice(_TIMER_CONTEXT_ADJUST_HISTORY).format(minutes=minutes),
        "operation": operation,
        "call": {"name": _TIMER_CONTEXT_TOOLS[operation], "arguments": {"minutes": minutes}},
    }


def _timer_context_bare(rng: random.Random) -> dict[str, Any]:
    operation = rng.choice(tuple(_TIMER_CONTEXT_BARE))
    amount = rng.choice((2, 3, 4, 5))
    arguments = {"minutes": amount} if operation in {"increase", "decrease"} else {}
    return {
        "operation": operation,
        "utterance": rng.choice(_TIMER_CONTEXT_BARE[operation]).format(minutes=amount),
        "call": {"name": _TIMER_CONTEXT_TOOLS[operation], "arguments": arguments},
    }


def _timer_context_plan(variant: str, rng: random.Random) -> dict[str, Any]:
    first, second = rng.sample(_TIMER_CONTEXT_NAMES, 2)
    minutes = rng.choice(_TIMER_CONTEXT_MINUTES)
    if variant == "bare_command":
        history = [_timer_context_start(rng, rng.choice((first, None)), minutes)]
        return {"variant": variant, "history": history, **_timer_context_bare(rng), "outcome": "ok"}
    if variant == "ambiguous_timers":
        history = [
            _timer_context_start(rng, first, minutes),
            _timer_context_start(rng, second, rng.choice(_TIMER_CONTEXT_MINUTES)),
        ]
        return {
            "variant": variant,
            "history": history,
            **_timer_context_bare(rng),
            "outcome": "multiple",
            "clarify_names": [f"{first} timer", f"{second} timer"],
        }
    if variant == "correction":
        if rng.random() < 0.5:
            history = [_timer_context_start(rng, rng.choice((first, None)), 10)]
            return {
                "variant": variant,
                "history": history,
                "operation": "increase",
                "utterance": rng.choice(_TIMER_CONTEXT_RAISE),
                "call": {"name": "HassIncreaseTimer", "arguments": {"minutes": 5}},
                "outcome": "ok",
            }
        history = [
            _timer_context_start(rng, rng.choice((first, None)), 10),
            _timer_context_adjust_history(rng, "increase", 10),
        ]
        return {
            "variant": variant,
            "history": history,
            "operation": "decrease",
            "utterance": rng.choice(_TIMER_CONTEXT_LOWER),
            "call": {"name": "HassDecreaseTimer", "arguments": {"minutes": 5}},
            "outcome": "ok",
        }
    history = [_timer_context_start(rng, first, minutes)]
    operation = rng.choice(tuple(_TIMER_CONTEXT_NAMED))
    amount = rng.choice((2, 3, 4, 5))
    utterance = rng.choice(_TIMER_CONTEXT_NAMED[operation]).format(name=second, minutes=amount)
    arguments = {"name": second, **({"minutes": amount} if operation in {"increase", "decrease"} else {})}
    return {
        "variant": variant,
        "history": history,
        "operation": operation,
        "utterance": utterance,
        "call": {"name": _TIMER_CONTEXT_TOOLS[operation], "arguments": arguments},
        "outcome": "not_found",
        "missing_name": second,
    }


def _timer_context_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    attempt: int,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    casing_counts: dict[str, Counter[str]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    plan = _timer_context_plan(slot["operation"], rng)
    home = generate_home(
        slot["index"] + attempt * 10000,
        slot["home_size"],
        rng,
        bare_name_rate=config.bare_name_rate,
    )
    call = plan["call"]
    spec = {
        "candidate_id": f"timer_context_{plan['variant']}_{slot['index']}",
        "seed": config.seed,
        "category": slot.get("robustness") or "ambiguity",
        "subcategory": call["name"],
        "home": home,
        "expected": {"kind": "action", "calls": [call]},
        "target_names": [],
        "spoken_targets": {},
        "excluded_names": [],
        "exclusion_scope": None,
        "contrastive_group": None,
        "phrasing_seed": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": plan["utterance"],
        "capability": "timers",
        "operation": plan["operation"],
        "tier": CAPABILITIES["timers"].tier,
        "semantic_id": f"timer_context_{plan['variant']}_{slot['index']}",
        "family": "timer_context",
        "removed_tools": [],
        "full_tool_catalog": True,
        "timer_context": {key: value for key, value in plan.items() if key not in {"utterance", "call", "operation"}},
    }
    turns = spec["timer_context"]["history"]
    turns[0]["utterance"] = finalize_training_utterance(
        turns[0]["utterance"], rng, upper=_balanced_upper(casing_counts, spec)
    )
    if plan["utterance"].casefold().startswith("no,"):
        # A correction only reads naturally as "no, ..." so it must not get a politeness wrapper in front of "no,".
        spec["utterance"] = _casing_only_text(plan["utterance"], rng, upper=_balanced_upper(casing_counts, spec))
    else:
        spec["utterance"] = finalize_training_utterance(plan["utterance"], rng)
    user_texts = [turn["utterance"] for turn in turns] + [spec["utterance"]]
    if any(text.casefold() in excluded for text in user_texts):
        return None, "excluded_prompt"
    if any(check_quality_eval_overlap(text) for text in user_texts):
        return None, "quality_eval_overlap"
    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )
    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)
    row["metadata"]["real_home"] = False
    row["metadata"]["grounding_family"] = None
    row["metadata"]["discrimination"] = False
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason:
        return None, reason
    dup_tracker.record(spec)
    return row, None


_RELATIVE_BRIGHTNESS_DELTAS = (-50, -40, -30, -25, -20, -10, 10, 20, 25, 30, 40, 50)
_RELATIVE_BRIGHTNESS_UP = (
    "turn {name} up {amount} percent", "brighten {name} by {amount} percent", "raise {name} by {amount} percent",
)
_RELATIVE_BRIGHTNESS_DOWN = (
    "dim {name} by {amount} percent", "turn {name} down {amount} percent", "lower {name} by {amount} percent",
)
_RELATIVE_BRIGHTNESS_READ = ("how bright is {name}", "what brightness is {name} at")


def _relative_brightness_plan(variant: str, name: str, rng: random.Random) -> dict[str, Any]:
    raw = rng.randrange(256)
    delta = rng.choice(_RELATIVE_BRIGHTNESS_DELTAS)
    templates = _RELATIVE_BRIGHTNESS_UP if delta > 0 else _RELATIVE_BRIGHTNESS_DOWN
    utterance = rng.choice(templates).format(name=name, amount=abs(delta))
    read_call = {"name": "GetLiveContext", "arguments": {"name": name, "domain": "light"}}
    if variant == "read_first":
        return {
            "variant": variant,
            "raw_brightness": raw,
            "history": [],
            "utterance": utterance,
            "call": read_call,
            "operation": "query_state",
        }
    # The label is computed from the brightness_pct the model reads back, as HA formats it.
    percent = ha_brightness_pct(raw)
    target = min(100, max(0, percent + delta))
    return {
        "variant": variant,
        "raw_brightness": raw,
        "history": [{"utterance": rng.choice(_RELATIVE_BRIGHTNESS_READ).format(name=name), "call": read_call}],
        "utterance": utterance,
        "call": {"name": "HassLightSet", "arguments": {"name": name, "domain": ["light"], "brightness": target}},
        "operation": "set_brightness",
    }


_LEVEL_ARGUMENTS = frozenset({"brightness", "volume_level"})


def _spec_gate(spec: dict[str, Any]) -> str | None:
    """Run the spec and utterance checks that can see the label.

    `validate_utterance` wants every numeric call argument spelled in the utterance, but the
    target level is derived from history, and it wants status labels to be query-shaped,
    which "turn X up 20 percent" is not. Both checks run on a proxy: level arguments are dropped
    from the calls and status rows are probed with a query. `validate_spec` still sees the real spec.
    """
    reason = validate_spec(spec)
    if reason:
        return reason
    expected = spec["expected"]
    calls = [
        {"name": call["name"], "arguments": {k: v for k, v in call["arguments"].items() if k not in _LEVEL_ARGUMENTS}}
        for call in expected["calls"]
    ]
    probe = spec["utterance"] if expected["kind"] == "action" else f"what is {spec['target_names'][0]}?"
    return validate_utterance({**spec, "expected": {**expected, "calls": calls}, "utterance": probe})


def _relative_brightness_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    attempt: int,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    casing_counts: dict[str, Counter[str]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    home = generate_home(
        slot["index"] + attempt * 10000,
        slot["home_size"],
        rng,
        bare_name_rate=config.bare_name_rate,
    )
    lights = [entity for entity in home["entities"] if entity["capability"] == "lights"]
    if not lights:
        return None, "relative_brightness_no_light"
    light = rng.choice(lights)
    plan = _relative_brightness_plan(slot["operation"], light["name"], rng)
    light["state"] = "on"
    light["capabilities"] = list(dict.fromkeys([*(light.get("capabilities") or []), "brightness"]))
    light["attributes"] = {"brightness": str(plan["raw_brightness"])}
    call = plan["call"]
    if call["name"] == "GetLiveContext":
        expected = {"kind": "status", "calls": [call], "state": "on"}
    else:
        expected = {"kind": "action", "calls": [call]}
    spec = {
        "candidate_id": f"relative_brightness_{plan['variant']}_{slot['index']}",
        "seed": config.seed,
        "category": slot.get("robustness") or "ordinary",
        "subcategory": call["name"],
        "home": home,
        "expected": expected,
        "target_names": [light["name"]],
        "spoken_targets": {},
        "excluded_names": [],
        "exclusion_scope": None,
        "contrastive_group": None,
        "phrasing_seed": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": plan["utterance"],
        "capability": "lights",
        "operation": plan["operation"],
        "tier": CAPABILITIES["lights"].tier,
        "semantic_id": f"relative_brightness_{plan['variant']}_{slot['index']}",
        "family": "relative_brightness",
        "removed_tools": [],
        "full_tool_catalog": True,
        "relative_brightness": {
            "variant": plan["variant"],
            "history": plan["history"],
            "raw_brightness": plan["raw_brightness"],
        },
    }
    for turn in spec["relative_brightness"]["history"]:
        turn["utterance"] = finalize_training_utterance(
            turn["utterance"], rng, upper=_balanced_upper(casing_counts, spec)
        )
    spec["utterance"] = finalize_training_utterance(plan["utterance"], rng)
    user_texts = [turn["utterance"] for turn in spec["relative_brightness"]["history"]] + [spec["utterance"]]
    if any(text.casefold() in excluded for text in user_texts):
        return None, "excluded_prompt"
    if any(check_quality_eval_overlap(text) for text in user_texts):
        return None, "quality_eval_overlap"
    reason = _spec_gate(spec)
    if reason:
        return None, reason
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )
    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)
    row["metadata"]["real_home"] = False
    row["metadata"]["grounding_family"] = None
    row["metadata"]["discrimination"] = False
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason:
        return None, reason
    dup_tracker.record(spec)
    return row, None


_CONVERSATION_CONTEXT_TOOLS = {"turn_on": "HassTurnOn", "turn_off": "HassTurnOff"}
_CONVERSATION_CONTEXT_HISTORY = {
    "turn_on": ("switch the {name} on", "turn the {name} on"),
    "turn_off": ("switch the {name} off", "turn the {name} off"),
}
_CONVERSATION_CONTEXT_PRONOUN = {
    "turn_on": ("switch it back on", "turn it back on", "turn that one on"),
    "turn_off": ("shut it off", "turn that off", "switch that off"),
}
_CONVERSATION_CONTEXT_OTHER = ("the other one too", "now the other one", "and the other one")
_CONVERSATION_CONTEXT_CORRECT = ("no, I meant the {name}", "no, I meant {name}")


def _conversation_context_targets(
    variant: str, entities: list[dict[str, Any]], rng: random.Random
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return (device the history acts on, device the trained turn acts on)."""
    if variant == "previous_turn_reference":
        if not entities:
            return None
        entity = rng.choice(entities)
        return entity, entity
    pairs = list(combinations(entities, 2))
    if variant == "other_one":
        # "the other one" only names a device when the area holds exactly two candidates; with three or more it is ambiguous.
        area_counts = Counter(entity["area"] for entity in entities)
        pairs = [
            (first, second) for first, second in pairs
            if first["area"] == second["area"] and area_counts[first["area"]] == 2
        ]
    if not pairs:
        return None
    first, second = rng.choice(pairs)
    return (first, second) if rng.random() < 0.5 else (second, first)


def _conversation_context_plan(variant: str, history_entity: dict[str, Any], target: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    operation = rng.choice(tuple(_CONVERSATION_CONTEXT_TOOLS))
    if variant == "previous_turn_reference":
        # The pronoun should read as a reversal of the last action: "turn it off" after "turn on", "switch it back on" after "turn off".
        history_operation = "turn_off" if operation == "turn_on" else "turn_on"
        utterance = rng.choice(_CONVERSATION_CONTEXT_PRONOUN[operation])
    elif variant == "other_one":
        history_operation = operation
        utterance = rng.choice(_CONVERSATION_CONTEXT_OTHER)
    else:
        history_operation = operation
        utterance = rng.choice(_CONVERSATION_CONTEXT_CORRECT).format(name=target["name"])
    history = [{
        "utterance": rng.choice(_CONVERSATION_CONTEXT_HISTORY[history_operation]).format(name=history_entity["name"]),
        "operation": history_operation,
        "call": _conversation_context_call(history_operation, history_entity),
    }]
    return {
        "variant": variant,
        "history": history,
        "utterance": utterance,
        "operation": operation,
        "call": _conversation_context_call(operation, target),
    }


def _conversation_context_call(operation: str, entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": _CONVERSATION_CONTEXT_TOOLS[operation], "arguments": {"name": entity["name"]}}


def _conversation_context_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    attempt: int,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    casing_counts: dict[str, Counter[str]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    home = generate_home(
        slot["index"] + attempt * 10000,
        slot["home_size"],
        rng,
        bare_name_rate=config.bare_name_rate,
    )
    entities = [entity for entity in home["entities"] if entity["capability"] == slot["capability"]]
    targets = _conversation_context_targets(slot["operation"], entities, rng)
    if targets is None:
        return None, "conversation_context_no_device_pair"
    plan = _conversation_context_plan(slot["operation"], *targets, rng)
    call = plan["call"]
    spec = {
        "candidate_id": f"conversation_context_{plan['variant']}_{slot['index']}",
        "seed": config.seed,
        "category": slot.get("robustness") or "ambiguity",
        "subcategory": call["name"],
        "home": home,
        "expected": {"kind": "action", "calls": [call]},
        "target_names": [],
        "spoken_targets": {},
        "excluded_names": [],
        "exclusion_scope": None,
        "contrastive_group": None,
        "phrasing_seed": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": plan["utterance"],
        "capability": slot["capability"],
        "operation": plan["operation"],
        "tier": CAPABILITIES[slot["capability"]].tier,
        "semantic_id": f"conversation_context_{plan['variant']}_{slot['index']}",
        "family": "conversation_context",
        "removed_tools": [],
        "full_tool_catalog": True,
        "conversation_context": {"variant": plan["variant"], "history": plan["history"]},
    }
    for turn in spec["conversation_context"]["history"]:
        turn["utterance"] = finalize_training_utterance(
            turn["utterance"], rng, upper=_balanced_upper(casing_counts, spec)
        )
    spec["utterance"] = _casing_only_text(plan["utterance"], rng, upper=_balanced_upper(casing_counts, spec))
    user_texts = [turn["utterance"] for turn in spec["conversation_context"]["history"]] + [spec["utterance"]]
    if any(text.casefold() in excluded for text in user_texts):
        return None, "excluded_prompt"
    if any(check_quality_eval_overlap(text) for text in user_texts):
        return None, "quality_eval_overlap"
    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )
    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)
    row["metadata"]["real_home"] = False
    row["metadata"]["grounding_family"] = None
    row["metadata"]["discrimination"] = False
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason:
        return None, reason
    dup_tracker.record(spec)
    return row, None


_RESTORE_READ = {
    "lights": ("how bright is {name}", "what brightness is {name} at"),
    "media_players": ("what volume is {name} at", "how loud is {name}"),
}
_RESTORE_CHANGE = {
    "lights": ("set {name} to {amount} percent", "make {name} {amount} percent bright"),
    "media_players": ("set the {name} volume to {amount}", "turn the {name} volume to {amount} percent"),
}
_RESTORE_BACK = ("put it back", "set it back to how it was", "put {name} back the way it was", "set {name} back to how it was")
_RESTORE_DOMAINS = {"lights": "light", "media_players": "media_player"}


def _restore_plan(capability: str, operation: str, name: str, rng: random.Random) -> dict[str, Any]:
    """Read the level, change it, then the trained turn returns to the level that was read."""
    if operation == "set_brightness":
        raw = rng.randrange(26, 256)
        attributes = {"brightness": str(raw)}
        earlier = ha_brightness_pct(raw)
    else:
        volume = rng.randrange(10, 81)
        attributes = {"volume_level": str(volume / 100)}
        earlier = round(float(attributes["volume_level"]) * 100)
    changed = rng.choice([level for level in range(10, 101) if level != earlier])
    if operation == "set_brightness":
        change_call = {"name": "HassLightSet", "arguments": {"name": name, "domain": ["light"], "brightness": changed}}
        restore_call = {"name": "HassLightSet", "arguments": {"name": name, "domain": ["light"], "brightness": earlier}}
    else:
        change_call = {"name": "HassSetVolume", "arguments": {"name": name, "domain": ["media_player"], "volume_level": changed}}
        restore_call = {"name": "HassSetVolume", "arguments": {"name": name, "domain": ["media_player"], "volume_level": earlier}}
    read_call = {"name": "GetLiveContext", "arguments": {"name": name, "domain": _RESTORE_DOMAINS[capability]}}
    return {
        "attributes": attributes,
        "earlier": earlier,
        "history": [
            {"utterance": rng.choice(_RESTORE_READ[capability]).format(name=name), "call": read_call},
            {"utterance": rng.choice(_RESTORE_CHANGE[capability]).format(name=name, amount=changed), "call": change_call},
        ],
        "utterance": rng.choice(_RESTORE_BACK).format(name=name),
        "call": restore_call,
    }


def _finish_context_row(
    spec: dict[str, Any],
    config: GeneratorConfig,
    *,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    user_texts: list[str],
    gate: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    if any(text.casefold() in excluded for text in user_texts):
        return None, "excluded_prompt"
    if any(check_quality_eval_overlap(text) for text in user_texts):
        return None, "quality_eval_overlap"
    reason = gate(spec)
    if reason:
        return None, reason
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )
    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)
    row["metadata"]["real_home"] = False
    row["metadata"]["grounding_family"] = None
    row["metadata"]["discrimination"] = False
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason:
        return None, reason
    dup_tracker.record(spec)
    return row, None


def _restore_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    attempt: int,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    casing_counts: dict[str, Counter[str]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    home = generate_home(
        slot["index"] + attempt * 10000,
        slot["home_size"],
        rng,
        bare_name_rate=config.bare_name_rate,
    )
    capability, operation = slot["capability"], slot["operation"]
    candidates = [entity for entity in home["entities"] if entity["capability"] == capability]
    if not candidates:
        return None, "restore_no_target"
    entity = rng.choice(candidates)
    plan = _restore_plan(capability, operation, entity["name"], rng)
    feature = "brightness" if operation == "set_brightness" else "volume"
    entity["state"] = "playing" if capability == "media_players" else "on"
    entity["capabilities"] = list(dict.fromkeys([*(entity.get("capabilities") or []), feature]))
    entity["attributes"] = plan["attributes"]
    name = entity["name"]
    call = plan["call"]
    spec = {
        "candidate_id": f"restore_{operation}_{slot['index']}",
        "seed": config.seed,
        "category": slot.get("robustness") or "ordinary",
        "subcategory": call["name"],
        "home": home,
        "expected": {"kind": "action", "calls": [call]},
        # The trained turn only names the device sometimes ("put it back" does not), so the
        # target is only checked when the utterance itself carries it.
        "target_names": [name] if name.casefold() in plan["utterance"].casefold() else [],
        "spoken_targets": {},
        "excluded_names": [],
        "exclusion_scope": None,
        "contrastive_group": None,
        "phrasing_seed": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": plan["utterance"],
        "capability": capability,
        "operation": operation,
        "tier": CAPABILITIES[capability].tier,
        "semantic_id": f"restore_{operation}_{slot['index']}",
        "family": "restore",
        "removed_tools": [],
        "full_tool_catalog": True,
        "restore": {"history": plan["history"], "earlier": plan["earlier"]},
    }
    read, change = spec["restore"]["history"]
    # "what brightness is X at" would read as "could you what brightness is X at" under finalize_training_utterance.
    read["utterance"] = _casing_only_text(read["utterance"], rng, upper=_balanced_upper(casing_counts, spec))
    change["utterance"] = finalize_training_utterance(change["utterance"], rng, upper=_balanced_upper(casing_counts, spec))
    spec["utterance"] = finalize_training_utterance(plan["utterance"], rng)
    user_texts = [read["utterance"], change["utterance"], spec["utterance"]]
    return _finish_context_row(
        spec, config, excluded=excluded, dup_tracker=dup_tracker, user_texts=user_texts, gate=_spec_gate,
    )


_CONDITIONAL_TEMPLATES = {
    "on": ("if {name} is on, turn it off", "turn {name} off if it's on"),
    "off": ("if {name} is off, turn it on", "turn {name} on if it's off"),
}
_CONDITIONAL_READ = ("is {name} on", "is {name} currently on", "check whether {name} is on")


def _conditional_plan(variant: str, entity: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    condition = rng.choice(("on", "off"))
    name = entity["name"]
    # The action that runs when the condition holds is the opposite of the condition.
    action = "turn_off" if condition == "on" else "turn_on"
    if variant == "condition_holds":
        state = condition
    elif variant == "condition_fails":
        state = "off" if condition == "on" else "on"
    else:
        state = rng.choice(("on", "off"))
    read_call = {"name": "GetLiveContext", "arguments": {"name": name, "domain": entity["domain"]}}
    history = [] if variant == "read_first" else [{
        "utterance": rng.choice(_CONDITIONAL_READ).format(name=name),
        "call": read_call,
    }]
    return {
        "variant": variant,
        "condition": condition,
        "state": state,
        "history": history,
        "utterance": rng.choice(_CONDITIONAL_TEMPLATES[condition]).format(name=name),
        "read_call": read_call,
        "action_call": build_turn_off(entity) if action == "turn_off" else build_turn_on(entity),
        "action": action,
    }


def _conditional_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    attempt: int,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    casing_counts: dict[str, Counter[str]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    home = generate_home(
        slot["index"] + attempt * 10000,
        slot["home_size"],
        rng,
        bare_name_rate=config.bare_name_rate,
    )
    capability = slot["capability"]
    candidates = [entity for entity in home["entities"] if entity["capability"] == capability]
    if not candidates:
        return None, "conditional_no_target"
    entity = rng.choice(candidates)
    variant = slot["operation"]
    plan = _conditional_plan(variant, entity, rng)
    entity["state"] = plan["state"]
    if variant == "read_first":
        expected = {"kind": "status", "calls": [plan["read_call"]], "state": plan["state"]}
        trained_call = plan["read_call"]
        operation = "query_state"
    elif variant == "condition_holds":
        expected = {"kind": "action", "calls": [plan["action_call"]]}
        trained_call = plan["action_call"]
        operation = plan["action"]
    else:
        # No eval response type covers "already in that state", so this branch is trained but not scored.
        expected = {"kind": "no_action", "response": "already_in_state", "calls": []}
        trained_call = plan["action_call"]
        operation = plan["action"]
    name = entity["name"]
    spec = {
        "candidate_id": f"conditional_{variant}_{slot['index']}",
        "seed": config.seed,
        "category": slot.get("robustness") or "ordinary",
        "subcategory": trained_call["name"],
        "home": home,
        "expected": expected,
        "target_names": [name],
        "spoken_targets": {},
        "excluded_names": [],
        "exclusion_scope": None,
        "contrastive_group": None,
        "phrasing_seed": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": plan["utterance"],
        "capability": capability,
        "operation": operation,
        "tier": CAPABILITIES[capability].tier,
        "semantic_id": f"conditional_{variant}_{slot['index']}",
        "family": "conditional",
        "removed_tools": [],
        "full_tool_catalog": True,
        "conditional": {
            "variant": variant,
            "condition": plan["condition"],
            "state": plan["state"],
            "history": plan["history"],
        },
    }
    for turn in spec["conditional"]["history"]:
        turn["utterance"] = _casing_only_text(turn["utterance"], rng, upper=_balanced_upper(casing_counts, spec))
    # "if ..." and "is ... on" would read as "could you if ..." under finalize_training_utterance.
    spec["utterance"] = _casing_only_text(plan["utterance"], rng, upper=_balanced_upper(casing_counts, spec))
    user_texts = [turn["utterance"] for turn in spec["conditional"]["history"]] + [spec["utterance"]]
    return _finish_context_row(
        spec, config, excluded=excluded, dup_tracker=dup_tracker, user_texts=user_texts, gate=_spec_gate,
    )
