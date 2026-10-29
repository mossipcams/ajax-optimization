
from __future__ import annotations

import hashlib
import json
import random
import re
import zlib
from datetime import datetime, timedelta
from typing import Any

from generators.capability_registry import operation_spec
from generators.context import _dump, area_context_for, system_prompt
from generators.planning import CORE_TOOL_NAMES
from generators.tools import namespaced_tool_name, production_catalog, script_tools


def scenario_to_spec(scenario: dict[str, Any]) -> dict[str, Any]:
    from generators.gold import target_names_from_expected

    expected = scenario["expected"]
    return {
        "candidate_id": scenario.get("semantic_id", f"candidate_{scenario.get('scenario_index', 0):06d}"),
        "seed": scenario.get("seed", 0),
        "category": scenario.get("robustness", "ordinary"),
        "subcategory": scenario.get("operation", ""),
        "home": scenario["home"],
        "expected": expected,
        "target_names": target_names_from_expected(expected),
        "spoken_targets": scenario.get("spoken_targets", {}),
        "excluded_names": scenario.get("excluded_names", []),
        "exclusion_scope": scenario.get("exclusion_scope"),
        "contrastive_group": scenario.get("contrastive_group"),
        "phrasing_seed": scenario.get("phrasing_seed"),
        "request_hint": scenario.get("request_hint", ""),
        "stt_corruption": scenario.get("stt_corruption"),
        "utterance": scenario.get("utterance"),
        "capability": scenario.get("capability"),
        "operation": scenario.get("operation"),
        "tier": scenario.get("tier"),
        "semantic_id": scenario.get("semantic_id"),
        "family": scenario.get("family"),
        "removed_tools": scenario.get("removed_tools", []),
    }


DECOY_WITHHOLD_RATE = 0.3


def _offered_catalog(spec: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    removed = list(spec["expected"].get("unavailable_tools") or spec.get("removed_tools") or [])
    removed_ns = [namespaced_tool_name(t) if "__" not in t else t for t in removed]
    if not removed_ns:
        removed_ns = _decoy_removals(spec, messages)
    return production_catalog(spec["home"], removed_tools=removed_ns)


def _decoy_removals(spec: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
    rng = random.Random(zlib.crc32(f"decoy:{spec['candidate_id']}".encode()))
    if rng.random() >= DECOY_WITHHOLD_RATE:
        return []
    needed = {call["function"]["name"] for m in messages for call in m.get("tool_calls") or []}
    op = operation_spec(spec.get("capability") or "", spec.get("operation") or "")
    if op and op.tool_name:
        needed.add(namespaced_tool_name(op.tool_name))
    for call in (spec["expected"].get("requested") or {}).get("calls") or []:
        needed.add(namespaced_tool_name(call["name"]))
    core = {namespaced_tool_name(name) for name in CORE_TOOL_NAMES}
    candidates = [
        t["function"]["name"]
        for t in production_catalog(spec["home"])
        if t["function"]["name"] not in needed and t["function"]["name"] not in core
    ]
    return rng.sample(candidates, min(len(candidates), rng.choice((1, 1, 2))))


def _call_id(candidate_id: str, index: int, call: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}:{index}:{json.dumps(call, sort_keys=True)}".encode()
    ).hexdigest()[:12]
    return f"call_{digest}"


def _did_you_mean(names: list[str]) -> str:
    labels = [
        name if name.casefold().startswith("the ") else f"the {name}"
        for name in names[:3]
        if name
    ]
    if len(labels) == 1:
        return f"Did you mean {labels[0]}?"
    if len(labels) == 2:
        return f"Did you mean {labels[0]} or {labels[1]}?"
    if len(labels) >= 3:
        return f"Did you mean {labels[0]}, {labels[1]}, or {labels[2]}?"
    return ""


def _final_text(spec: dict[str, Any]) -> str:
    expected = spec["expected"]
    if expected["kind"] == "status":
        names = spec.get("target_names") or []
        from generators.gold import target_names_from_expected

        names = names or target_names_from_expected(expected)
        state = str(expected.get("state", "unknown"))
        if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d.*", state):
            return f"{names[0]} was last activated at {state[11:16]}."
        if state.isdigit():
            return f"{names[0]} has {state} item{'' if state == '1' else 's'}."
        return f"{names[0]} is {state}."
    if expected["kind"] == "action":
        calls = expected.get("calls") or []
        if len(calls) == 1 and calls[0]["name"] in RUNTIME_ANSWER_TOOLS:
            return _runtime_answer(spec, calls[0])
        return "Done."
    if expected.get("response") == "clarify":
        named = _did_you_mean(list(expected.get("candidates") or []))
        if named:
            return named
        if spec.get("capability") == "scripts":
            return "Which routine did you mean?"
        return "Which device did you mean?"
    if expected.get("response") == "device_unsupported":
        names = expected.get("unsupported_names") or []
        return f"{names[0]} does not support that." if names else "That device does not support that."
    if expected.get("subfamily") == "unexposed_domain":
        name = expected["unavailable"]["name"]
        return f"I can't control {name}. It is only available as context."
    if expected.get("response") == "device_absent":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "this area")
        device = unavailable.get("type", "device")
        return f"There is no {device} in the {area}."
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "this area").casefold()
        device_type = unavailable.get("type", "devices")
        return f"The {area} has no {device_type} available."
    return {
        "not_understood": "Sorry, I didn't catch that.",
        "clarify": "Which device did you mean?",
        "unsupported": "I can't do that with the available Home Assistant tools.",
        "refuse": "I can't help with that request.",
    }.get(expected.get("response", ""), "I can't help with that request.")


def _production_metadata(spec: dict[str, Any], offered: list[dict[str, Any]], utterance: str) -> dict[str, Any]:
    return {
        "candidate_id": spec["candidate_id"],
        "semantic_id": spec.get("semantic_id", spec["candidate_id"]),
        "template_family": spec.get("contrastive_group") or spec.get("category", "ordinary"),
        "phrasing_family": spec.get("subcategory", ""),
        "seed": spec.get("semantic_id", spec["candidate_id"]),
        "generation_seed": spec.get("seed"),
        "category": spec.get("category"),
        "subcategory": spec.get("subcategory"),
        "capability": spec.get("capability"),
        "operation": spec.get("operation"),
        "tier": spec.get("tier"),
        "family": spec.get("family"),
        "home_id": spec["home"]["home_id"],
        "home_size": spec["home"].get("size"),
        "expected_target_names": spec.get("target_names", []),
        "contrastive_group": spec.get("contrastive_group"),
        "stt_corruption": spec.get("stt_corruption"),
        "paraphrase_source": spec.get("paraphrase_source"),
        "excluded_names": spec.get("excluded_names", []),
        "spoken_targets": spec.get("spoken_targets", {}),
        "no_action_reason": spec["expected"].get("response"),
        "unavailable": spec["expected"].get("unavailable"),
        "unavailable_tools": spec["expected"].get("unavailable_tools", []),
        "linguistics": spec.get("linguistics", []),
        "contract": "production_v3",
        "area_context": area_context_for(spec["home"], utterance.strip()).as_dict(),
        "offered_tool_count": len(offered),
        "catalog_source": "production_exposure",
        "full_tool_catalog": spec.get("full_tool_catalog", True),
    }


def ha_brightness_pct(brightness: int) -> int:
    """HA's light `brightness_pct`: rounded percent, but at least 1 for any lit light."""
    percent = round(brightness / 255 * 100)
    return max(percent, 1) if brightness > 0 else percent


def _live_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Shape entity attributes as HA's LLM live context renders them.

    Integer attributes are stringified, `brightness_pct` is derived from `brightness`, and
    `volume_level` stays a float.
    """
    shaped = dict(attributes)
    if "brightness" in shaped:
        brightness = int(shaped["brightness"])
        shaped["brightness"] = str(brightness)
        shaped["brightness_pct"] = str(ha_brightness_pct(brightness))
    if "volume_level" in shaped:
        shaped["volume_level"] = float(shaped["volume_level"])
    return shaped


def _home_entities_by_name(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for entity in spec["home"].get("entities", []):
        by_name.setdefault(entity["name"], entity)
        for alias in entity.get("aliases", []):
            by_name.setdefault(alias, entity)
    return by_name


def _live_context_entries(spec: dict[str, Any], call: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = call.get("arguments") or {}
    by_name = _home_entities_by_name(spec)
    wanted = [arguments.get("name"), *(spec.get("target_names") or [])]
    resolved: list[dict[str, Any]] = []
    for name in wanted:
        if not name:
            continue
        entity = by_name.get(name)
        if entity and all(item is not entity for item in resolved):
            resolved.append(entity)
    entries: list[dict[str, Any]] = []
    for entity in resolved:
        info: dict[str, Any] = {"names": entity["name"], "domain": entity["domain"]}
        info["state"] = str(entity.get("state") or spec["expected"].get("state", "unknown"))
        area = entity.get("area")
        if area and area != "Unassigned":
            info["areas"] = area
        attributes = entity.get("attributes")
        if attributes:
            info["attributes"] = _live_attributes(attributes)
        entries.append(info)
    if not entries:
        fallback: dict[str, Any] = {"names": next((name for name in wanted if name), "device")}
        domain = arguments.get("domain")
        if isinstance(domain, str) and domain:
            fallback["domain"] = domain
        fallback["state"] = str(spec["expected"].get("state", "unknown"))
        entries.append(fallback)
    return entries


def _intent_success(spec: dict[str, Any], call: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per targeted home entity. Timers and other non-entity names produce none."""
    arguments = call.get("arguments") or {}
    target = arguments.get("name")
    entity = _home_entities_by_name(spec).get(target) if target else None
    if entity is None:
        return []
    spoken = (spec.get("spoken_targets") or {}).get(target, target)
    return [{"name": spoken, "type": "entity", "id": entity["entity_id"]}]


RUNTIME_ANSWER_TOOLS = ("GetDateTime", "HassTimerStatus")


def _runtime_moment(spec: dict[str, Any]) -> datetime:
    rng = random.Random(zlib.crc32(f"datetime:{spec['candidate_id']}".encode()))
    return datetime(2026, 1, 1) + timedelta(seconds=rng.randrange(365 * 86400))


def _round_time(hours: int, minutes: int, seconds: int) -> tuple[int, int, int]:
    if hours > 0:
        rounded_hours = hours
        rounded_seconds = 0
        if minutes > 45:
            rounded_hours += 1
            rounded_minutes = 0
        elif minutes < 15:
            rounded_minutes = 0
        else:
            rounded_minutes = 30
    elif minutes > 0:
        rounded_hours = 0
        rounded_minutes = minutes
        if seconds > 45:
            rounded_minutes += 1
            rounded_seconds = 0
        elif seconds < 15:
            rounded_seconds = 0
        else:
            rounded_seconds = 30
    else:
        rounded_hours = 0
        rounded_minutes = 0
        if seconds > 50:
            rounded_minutes = 1
            rounded_seconds = 0
        elif seconds < 10:
            rounded_seconds = seconds
        else:
            rounded_seconds = seconds - (seconds % 10)
    return rounded_hours, rounded_minutes, rounded_seconds


def _timer_status_timer(spec: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(zlib.crc32(f"timer:{spec['candidate_id']}".encode()))
    start_minutes = rng.choice((5, 10, 15, 20, 30, 45))
    minutes_left = rng.randint(2, start_minutes - 1)
    seconds_left = rng.randint(2, 59)
    rounded_hours, rounded_minutes, rounded_seconds = _round_time(0, minutes_left, seconds_left)
    return {
        "id": "".join(rng.choice("0123456789ABCDEFGHJKMNPQRSTVWXYZ") for _ in range(26)),
        "name": (call.get("arguments") or {}).get("name") or "",
        "device_id": format(rng.getrandbits(128), "032x"),
        "language": "en",
        "start_hours": 0,
        "start_minutes": start_minutes,
        "start_seconds": 0,
        "is_active": True,
        "hours_left": 0,
        "minutes_left": minutes_left,
        "seconds_left": seconds_left,
        "rounded_hours_left": rounded_hours,
        "rounded_minutes_left": rounded_minutes,
        "rounded_seconds_left": rounded_seconds,
        "total_seconds_left": minutes_left * 60 + seconds_left,
    }


def _runtime_answer(spec: dict[str, Any], call: dict[str, Any]) -> str:
    if call["name"] == "GetDateTime":
        moment = _runtime_moment(spec)
        meridiem = "AM" if moment.hour < 12 else "PM"
        return f"It's {moment.hour % 12 or 12}:{moment.minute:02d} {meridiem}."
    timer = _timer_status_timer(spec, call)
    where = f"the {timer['name']} timer" if timer["name"] else "your timer"
    return f"There are {timer['minutes_left']} minutes and {timer['seconds_left']} seconds left on {where}."


def _tool_result(spec: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    if call["name"] == "GetLiveContext":
        header = "Live Context: An overview of the areas and the devices in this smart home:"
        return {"result": f"{header}\n{_dump(_live_context_entries(spec, call))}"}
    if call["name"] == "GetDateTime":
        moment = _runtime_moment(spec)
        return {
            "date": moment.strftime("%Y-%m-%d"),
            "time": moment.strftime("%H:%M:%S"),
            "timezone": "CDT",
            "weekday": moment.strftime("%A"),
        }
    if call["name"] == "HassTimerStatus":
        return {
            "speech": {},
            "response_type": "action_done",
            "speech_slots": {"timers": [_timer_status_timer(spec, call)]},
            "data": {"success": [], "failed": []},
        }
    return {
        "speech": {},
        "response_type": "action_done",
        "data": {"success": _intent_success(spec, call), "failed": []},
    }


def _append_action_turn(
    messages: list[dict[str, Any]],
    spec: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    train_tool_turn: bool,
) -> None:
    tool_name = namespaced_tool_name
    rendered_calls = []
    call_ids = []
    for index, call in enumerate(calls):
        call_id = _call_id(spec["candidate_id"], index, call)
        call_ids.append(call_id)
        rendered_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name(call["name"]),
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True),
                },
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "train_on_turn": train_tool_turn,
            "tool_calls": rendered_calls,
        }
    )
    for call_id, call in zip(call_ids, calls):
        result = _tool_result(spec, call)
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "train_on_turn": False,
                "tool_call_id": call_id,
            }
        )
    messages.append({"role": "assistant", "content": _final_text(spec), "train_on_turn": True})


def _render_follow_up(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    follow_up_reply = spec.get("follow_up_reply")
    candidates = spec.get("follow_up_clarify_candidates") or []
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    if not isinstance(follow_up_reply, str) or not follow_up_reply.strip():
        raise ValueError("missing_follow_up_reply")
    if len(candidates) < 2:
        raise ValueError("missing_follow_up_candidates")
    calls = spec["expected"].get("calls") or []
    if not calls:
        raise ValueError("follow_up_requires_action_gold")
    clarify_text = _did_you_mean(list(candidates))
    if not clarify_text:
        raise ValueError("missing_clarify_text")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
        {"role": "assistant", "content": clarify_text, "train_on_turn": True},
        {"role": "user", "content": follow_up_reply.strip(), "train_on_turn": False},
    ]
    _append_action_turn(messages, spec, calls, train_tool_turn=True)
    offered = _offered_catalog(spec, messages)
    return {"messages": messages, "tools": offered, "metadata": _production_metadata(spec, offered, utterance)}


def _render_correction(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.correction import (
        format_synthetic_validation_error,
        validation_error_message,
        wrong_name_tool_call,
    )
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    wrong_name = spec.get("correction_wrong_name")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    if not isinstance(wrong_name, str) or not wrong_name.strip():
        raise ValueError("missing_correction_wrong_name")
    calls = spec["expected"].get("calls") or []
    if not calls:
        raise ValueError("correction_requires_action_gold")
    correct_call = calls[0]
    bad_call = wrong_name_tool_call(correct_call, wrong_name)
    tool_name = namespaced_tool_name
    planned = [{"tool_calls": [{"function": {"name": tool_name(call["name"])}} for call in calls]}]
    offered = _offered_catalog(spec, planned)
    allowed_tools = sorted({entry["function"]["name"] for entry in offered})
    wrong_id = _call_id(spec["candidate_id"], 0, bad_call)
    correct_id = _call_id(spec["candidate_id"], 1, correct_call)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
        {
            "role": "assistant",
            "content": "",
            "train_on_turn": False,
            "tool_calls": [
                {
                    "id": wrong_id,
                    "type": "function",
                    "function": {
                        "name": tool_name(bad_call["name"]),
                        "arguments": json.dumps(bad_call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(
                format_synthetic_validation_error(
                    code="schema_mismatch",
                    message=validation_error_message(wrong_name),
                    allowed_tools=allowed_tools,
                ),
                ensure_ascii=False,
            ),
            "train_on_turn": False,
            "tool_call_id": wrong_id,
        },
    ]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "train_on_turn": True,
            "tool_calls": [
                {
                    "id": correct_id,
                    "type": "function",
                    "function": {
                        "name": tool_name(correct_call["name"]),
                        "arguments": json.dumps(correct_call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(_tool_result(spec, correct_call), ensure_ascii=False),
            "train_on_turn": False,
            "tool_call_id": correct_id,
        }
    )
    messages.append({"role": "assistant", "content": _final_text(spec), "train_on_turn": True})
    return {"messages": messages, "tools": offered, "metadata": _production_metadata(spec, offered, utterance)}


def _tool_call_message(call_id: str, call: dict[str, Any], *, train_on_turn: bool) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "train_on_turn": train_on_turn,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": namespaced_tool_name(call["name"]),
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True),
                },
            }
        ],
    }


def _turn_messages(
    candidate_id: str,
    number: int,
    utterance: str,
    call: dict[str, Any],
    result: dict[str, Any],
    *,
    train_on_turn: bool,
) -> list[dict[str, Any]]:
    """One user request, its tool call, and the tool result; only the call may be trained."""
    call_id = _call_id(candidate_id, number, call)
    return [
        {"role": "user", "content": utterance, "train_on_turn": False},
        _tool_call_message(call_id, call, train_on_turn=train_on_turn),
        {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "train_on_turn": False,
            "tool_call_id": call_id,
        },
    ]


def _timer_context_result(spec: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    outcome = spec["timer_context"]["outcome"]
    if outcome == "multiple":
        return {"error": "MultipleTimersMatchedError", "error_text": "Multiple timers matched"}
    if outcome == "not_found":
        return {"error": "TimerNotFoundError", "error_text": "Timer not found"}
    return _tool_result(spec, call)


def _timer_context_reply(spec: dict[str, Any]) -> str:
    context = spec["timer_context"]
    if context["outcome"] == "multiple":
        return _did_you_mean(list(context["clarify_names"]))
    if context["outcome"] == "not_found":
        return f"There's no {context['missing_name']} timer running."
    return _final_text(spec)


def _render_timer_context(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    history = list((spec.get("timer_context") or {}).get("history") or [])
    if not 1 <= len(history) <= 2:
        raise ValueError("timer_context_history_size")
    calls = spec["expected"].get("calls") or []
    if len(calls) != 1:
        raise ValueError("timer_context_requires_one_call")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
    ]
    for number, turn in enumerate(history):
        call = turn["call"]
        turn_spec = {**spec, "operation": turn["operation"], "expected": {"kind": "action", "calls": [call]}}
        messages.extend(_turn_messages(
            spec["candidate_id"], number, turn["utterance"], call, _tool_result(turn_spec, call),
            train_on_turn=False,
        ))
    call = calls[0]
    messages.extend(_turn_messages(
        spec["candidate_id"], len(history), utterance.strip(), call, _timer_context_result(spec, call),
        train_on_turn=True,
    ))
    messages.append({"role": "assistant", "content": _timer_context_reply(spec), "train_on_turn": True})
    offered = _offered_catalog(spec, messages)
    metadata = _production_metadata(spec, offered, utterance)
    named = [turn["call"] for turn in history] + [call]
    metadata["expected_target_names"] = list(dict.fromkeys(
        str(item["arguments"]["name"]) for item in named if item["arguments"].get("name")
    ))
    return {"messages": messages, "tools": offered, "metadata": metadata}


def _render_relative_brightness(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    context = spec.get("relative_brightness") or {}
    history = list(context.get("history") or [])
    expected_turns = 1 if context.get("variant") == "set_from_context" else 0
    if len(history) != expected_turns:
        raise ValueError("relative_brightness_history_size")
    calls = spec["expected"].get("calls") or []
    if len(calls) != 1:
        raise ValueError("relative_brightness_requires_one_call")
    call = calls[0]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
    ]
    for number, turn in enumerate(history):
        read = turn["call"]
        messages.extend(_turn_messages(
            spec["candidate_id"], number, turn["utterance"], read, _tool_result(spec, read), train_on_turn=False,
        ))
    messages.extend(_turn_messages(
        spec["candidate_id"], len(history), utterance.strip(), call, _tool_result(spec, call), train_on_turn=True,
    ))
    if call["name"] == "GetLiveContext":
        reply = _final_text(spec)
    else:
        arguments = call["arguments"]
        reply = f"{arguments['name']} is now at {arguments['brightness']} percent."
    messages.append({"role": "assistant", "content": reply, "train_on_turn": True})
    offered = _offered_catalog(spec, messages)
    metadata = _production_metadata(spec, offered, utterance)
    metadata["expected_target_names"] = list(spec.get("target_names") or [])
    return {"messages": messages, "tools": offered, "metadata": metadata}


def _render_conversation_context(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    history = list((spec.get("conversation_context") or {}).get("history") or [])
    if len(history) != 1:
        raise ValueError("conversation_context_history_size")
    calls = spec["expected"].get("calls") or []
    if len(calls) != 1:
        raise ValueError("conversation_context_requires_one_call")
    call = calls[0]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
    ]
    turn = history[0]
    turn_spec = {**spec, "operation": turn["operation"], "expected": {"kind": "action", "calls": [turn["call"]]}}
    messages.extend(_turn_messages(
        spec["candidate_id"], 0, turn["utterance"], turn["call"], _tool_result(turn_spec, turn["call"]),
        train_on_turn=False,
    ))
    messages.extend(_turn_messages(
        spec["candidate_id"], 1, utterance.strip(), call, _tool_result(spec, call), train_on_turn=True,
    ))
    state = "on" if spec["operation"] == "turn_on" else "off"
    messages.append({
        "role": "assistant",
        "content": f"{call['arguments']['name']} is now {state}.",
        "train_on_turn": True,
    })
    offered = _offered_catalog(spec, messages)
    metadata = _production_metadata(spec, offered, utterance)
    metadata["expected_target_names"] = [call["arguments"]["name"]]
    return {"messages": messages, "tools": offered, "metadata": metadata}


def _render_restore(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    history = list((spec.get("restore") or {}).get("history") or [])
    if len(history) != 2:
        raise ValueError("restore_history_size")
    calls = spec["expected"].get("calls") or []
    if len(calls) != 1:
        raise ValueError("restore_requires_one_call")
    call = calls[0]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
    ]
    for number, turn in enumerate(history):
        messages.extend(_turn_messages(
            spec["candidate_id"], number, turn["utterance"], turn["call"], _tool_result(spec, turn["call"]),
            train_on_turn=False,
        ))
    messages.extend(_turn_messages(
        spec["candidate_id"], len(history), utterance.strip(), call, _tool_result(spec, call), train_on_turn=True,
    ))
    arguments = call["arguments"]
    level = arguments["brightness"] if "brightness" in arguments else arguments["volume_level"]
    messages.append({"role": "assistant", "content": f"{arguments['name']} is now at {level} percent.", "train_on_turn": True})
    offered = _offered_catalog(spec, messages)
    metadata = _production_metadata(spec, offered, utterance)
    named = [turn["call"] for turn in history] + [call]
    metadata["expected_target_names"] = list(dict.fromkeys(
        str(item["arguments"]["name"]) for item in named if item["arguments"].get("name")
    ))
    return {"messages": messages, "tools": offered, "metadata": metadata}


def _render_conditional(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    context = spec.get("conditional") or {}
    variant = context.get("variant")
    history = list(context.get("history") or [])
    if len(history) != (0 if variant == "read_first" else 1):
        raise ValueError("conditional_history_size")
    name = spec["target_names"][0]
    calls = spec["expected"].get("calls") or []
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
    ]
    for number, turn in enumerate(history):
        messages.extend(_turn_messages(
            spec["candidate_id"], number, turn["utterance"], turn["call"], _tool_result(spec, turn["call"]),
            train_on_turn=False,
        ))
    number = len(history)
    if variant == "read_first":
        if len(calls) != 1:
            raise ValueError("conditional_requires_one_call")
        messages.extend(_turn_messages(
            spec["candidate_id"], number, utterance.strip(), calls[0], _tool_result(spec, calls[0]), train_on_turn=True,
        ))
        messages.append({"role": "assistant", "content": _final_text(spec), "train_on_turn": True})
    elif variant == "condition_holds":
        if len(calls) != 1:
            raise ValueError("conditional_requires_one_call")
        call = calls[0]
        messages.extend(_turn_messages(
            spec["candidate_id"], number, utterance.strip(), call, _tool_result(spec, call), train_on_turn=True,
        ))
        after = "off" if call["name"] == "HassTurnOff" else "on"
        messages.append({"role": "assistant", "content": f"{name} is now {after}.", "train_on_turn": True})
    elif variant == "condition_fails":
        if calls:
            raise ValueError("conditional_no_action_has_calls")
        # No tool call is made, so the user turn is the only message before the spoken answer.
        messages.append({"role": "user", "content": utterance.strip(), "train_on_turn": False})
        messages.append({"role": "assistant", "content": f"{name} is already {context['state']}.", "train_on_turn": True})
    else:
        raise ValueError("conditional_variant")
    offered = _offered_catalog(spec, messages)
    metadata = _production_metadata(spec, offered, utterance)
    metadata["expected_target_names"] = [name]
    return {"messages": messages, "tools": offered, "metadata": metadata}


def render_example(spec: dict[str, Any]) -> dict[str, Any]:
    family = spec.get("family")
    if family == "follow_up":
        return _render_follow_up(spec)
    if family == "correction":
        return _render_correction(spec)
    if family == "timer_context":
        return _render_timer_context(spec)
    if family == "relative_brightness":
        return _render_relative_brightness(spec)
    if family == "conversation_context":
        return _render_conversation_context(spec)
    if family == "restore":
        return _render_restore(spec)
    if family == "conditional":
        return _render_conditional(spec)
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    calls = spec["expected"].get("calls") or []
    tool_name = namespaced_tool_name
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
    ]
    if calls:
        rendered_calls = []
        call_ids = []
        for index, call in enumerate(calls):
            call_id = _call_id(spec["candidate_id"], index, call)
            call_ids.append(call_id)
            rendered_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name(call["name"]),
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": True,
                "tool_calls": rendered_calls,
            }
        )
        for call_id, call in zip(call_ids, calls):
            result = _tool_result(spec, call)
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "train_on_turn": False,
                    "tool_call_id": call_id,
                }
            )
    messages.append({"role": "assistant", "content": _final_text(spec), "train_on_turn": True})

    offered = _offered_catalog(spec, messages)

    metadata = {
        "candidate_id": spec["candidate_id"],
        "semantic_id": spec.get("semantic_id", spec["candidate_id"]),
        "template_family": spec.get("contrastive_group") or spec.get("category", "ordinary"),
        "phrasing_family": spec.get("subcategory", ""),
        "seed": spec.get("semantic_id", spec["candidate_id"]),
        "generation_seed": spec.get("seed"),
        "category": spec.get("category"),
        "subcategory": spec.get("subcategory"),
        "capability": spec.get("capability"),
        "operation": spec.get("operation"),
        "tier": spec.get("tier"),
        "family": spec.get("family"),
        "home_id": spec["home"]["home_id"],
        "home_size": spec["home"].get("size"),
        "expected_target_names": spec.get("target_names", []),
        "contrastive_group": spec.get("contrastive_group"),
        "stt_corruption": spec.get("stt_corruption"),
        "paraphrase_source": spec.get("paraphrase_source"),
        "excluded_names": spec.get("excluded_names", []),
        "exclusion_scope": spec.get("exclusion_scope"),
        "spoken_targets": spec.get("spoken_targets", {}),
        "no_action_reason": spec["expected"].get("response"),
        "unavailable": spec["expected"].get("unavailable"),
        "unavailable_tools": spec["expected"].get("unavailable_tools", []),
        "linguistics": spec.get("linguistics", []),
        "contract": "production_v3",
        "area_context": area_context_for(spec["home"], utterance.strip()).as_dict(),
        "offered_tool_count": len(offered),
        "catalog_source": "production_exposure",
        "full_tool_catalog": spec.get("full_tool_catalog", True),
    }
    return {"messages": messages, "tools": offered, "metadata": metadata}
