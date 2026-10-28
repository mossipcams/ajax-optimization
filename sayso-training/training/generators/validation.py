
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any
from pathlib import Path

import yaml

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools
from generators.context import STATIC_CONTEXT_HEADER
from generators.tools import _OPERATION_TOOLS, HA_TOOL_NAMESPACES, namespaced_tool_name, script_tool_name
from generators.capability_registry import CAPABILITIES, SupportLevel, entity_supports, is_satellite_name
from generators.config import DEFAULT_TOKEN_BUDGET
from generators.gold import _type_label, refers_to
from generators.stt_noise import _int_to_words, utterance_contains_target

_BANNED = re.compile(r"<tool_call>|evals/cases/|tool_call_start", re.I)
_PLACEHOLDER_AREA = "unassigned"


def _normalize_prompt(text: str) -> str:
    prompt = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    prompt = re.sub(r"^(?:(?:please|can you|could you|tell me) )+", "", prompt)
    return re.sub(r"(?: for me)+$", "", prompt)


@lru_cache(maxsize=1)
def quality_eval_prompts() -> frozenset[str]:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    from evals.cases import excluded_train_utterances
    return frozenset(_normalize_prompt(prompt) for prompt in excluded_train_utterances())


def check_quality_eval_overlap(utterance: str) -> bool:
    return _normalize_prompt(utterance) in quality_eval_prompts()


def corrupt_spec(spec: dict[str, Any], field: str) -> dict[str, Any]:
    corrupted = dict(spec)
    expected = dict(corrupted.get("expected", {}))
    calls = list(expected.get("calls") or [])
    if calls and field in {"wrong_tool", "wrong_entity"}:
        calls[0] = dict(calls[0])
        if field == "wrong_tool":
            calls[0]["name"] = "NonexistentTool"
        else:
            calls[0]["arguments"] = {**calls[0].get("arguments", {}), "name": "Invented Device"}
        expected["calls"] = calls
        corrupted["expected"] = expected
    return corrupted

_TIMER_NAME_TOOLS = frozenset({
    "HassStartTimer", "HassPauseTimer", "HassUnpauseTimer", "HassCancelTimer",
    "HassIncreaseTimer", "HassDecreaseTimer", "HassTimerStatus",
})


def _entity_can_run(entity: dict[str, Any], tool: str) -> bool:
    capability = entity.get("capability")
    cap = CAPABILITIES.get(capability)
    if cap is None:
        return True
    candidates = [op for op in cap.operations if op.tool_name == tool]
    if not candidates:
        return True
    return any(entity_supports(entity, capability, op.name) for op in candidates)


def _is_satellite_name(name: str) -> bool:
    return is_satellite_name(name)


def is_satellite_contrast(call_names: list[str], excluded: list[str]) -> bool:
    if len(call_names) != 1 or len(excluded) != 1:
        return False
    return _is_satellite_name(call_names[0]) != _is_satellite_name(excluded[0])


def satellite_contrast_spoken(utterance: str, call_names: list[str], excluded: list[str]) -> bool:
    if not is_satellite_contrast(call_names, excluded):
        return False
    lowered = utterance.casefold()
    if _is_satellite_name(excluded[0]):
        return lowered.endswith(", not you")
    return lowered.endswith(f", not the {excluded[0].casefold()}")


def shared_alias_action_conflict(spec: dict[str, Any]) -> bool:
    if (spec.get("expected") or {}).get("kind") != "action":
        return False
    text = (spec.get("utterance") or "").casefold()
    entities = (spec.get("home") or {}).get("entities") or []
    for entity in entities:
        for alias in entity.get("aliases") or []:
            needle = alias.casefold()
            if not re.search(rf"\b{re.escape(needle)}\b", text):
                continue
            if any(
                other is not entity
                and other.get("area") == entity.get("area")
                and needle in {name.casefold() for name in other.get("aliases") or []}
                for other in entities
            ):
                return True
    return False


def validate_spec(spec: dict[str, Any]) -> str | None:
    if spec.get("shared_alias") and shared_alias_action_conflict(spec):
        return "shared_alias_action"
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    cap = CAPABILITIES.get(spec.get("capability"))
    operation = next((op for op in cap.operations if op.name == spec.get("operation")), None) if cap else None
    if expected.get("kind") == "no_action":
        if spec.get("capability") == "timers" and expected.get("response") != "unsupported":
            return "timer_inventory_refusal"
        if expected.get("response") == "unsupported" and operation and operation.support != SupportLevel.UNAVAILABLE:
            requested = expected.get("requested", {}).get("calls", [])
            withheld = set(expected.get("unavailable_tools", []))
            if not requested or any(call["name"] not in withheld for call in requested):
                return "unsupported_without_withheld_tool"
        if expected.get("response") == "device_unsupported":
            named = expected.get("unsupported_names") or []
            by_name = {e["name"]: e for e in spec.get("home", {}).get("entities", [])}
            context_names = {e["name"] for e in spec.get("home", {}).get("context_entities") or []}
            if not named:
                return "device_unsupported_without_names"
            for name in named:
                if name not in by_name:
                    if name in context_names:
                        continue
                    return "unknown_canonical_entity"
                if entity_supports(by_name[name], spec.get("capability"), spec.get("operation")):
                    return "contradictory_device_support"
    if any(call.get("name") == "GetLiveContext" for call in calls) and expected.get("kind") != "status":
        return "state_query_requires_status_label"
    if expected.get("kind") == "status" and not any(call.get("name") == "GetLiveContext" for call in calls):
        return "status_requires_state_query"
    entities = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "").casefold()
        domains = {cap.domain for cap in CAPABILITIES.values() if _type_label(cap.name) == unavailable.get("type")}
        if any(
            e.get("area", "").casefold() == area
            and (e.get("domain") in domains or unavailable.get("type") == "devices")
            for e in entities.values()
        ):
            return "contradictory_absence"
    call_names = [str((call.get("arguments") or {}).get("name") or "") for call in calls]
    contrast = is_satellite_contrast(call_names, spec.get("excluded_names") or [])
    if spec.get("category") in {"multi_action", "exclusion"} and len(calls) < 2 and not contrast:
        return "missing_multiple_actions"
    if spec.get("category") == "exclusion" and not spec.get("excluded_names"):
        return "missing_excluded_target"
    schemas = tool_schema_map(v2_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    script_names = {
        script_tool_name(entity)
        for entity in spec.get("home", {}).get("entities", [])
        if entity.get("domain") == "script"
    }
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
        if name in script_names:
            if arguments:
                return "script_tool_takes_no_arguments"
            if any(
                script_tool_name(e) == name and e["name"] in excluded
                for e in entities.values()
                if e.get("domain") == "script"
            ):
                return "excluded_entity_called"
            continue
        reason = validate_tool_arguments(name, arguments, schemas)
        if reason:
            return reason
        target = arguments.get("name")
        if target is not None and target not in entities and name not in _TIMER_NAME_TOOLS:
            return "unknown_canonical_entity"
        if target in excluded:
            return "excluded_entity_called"
        if target in entities and not _entity_can_run(entities[target], name):
            return "entity_lacks_capability"
    for canonical, spoken in (spec.get("spoken_targets") or {}).items():
        if canonical not in entities or not str(spoken).strip():
            return "invalid_spoken_target"
    return None


_STATUS_QUERY = re.compile(r"status|\bwhat|^\W*(?:please\s+)?(?:is|are|did|does|do|how)\b|\?\W*$")


def _names_one_clarify_candidate(spec: dict[str, Any], lowered: str) -> bool:
    expected = spec.get("expected") or {}
    if spec.get("phrasing_seed"):
        return False
    if expected.get("response") == "clarify":
        candidates = list(expected.get("candidates") or [])
    else:
        candidates = list(spec.get("follow_up_clarify_candidates") or [])
    if len(candidates) < 2:
        return False
    entities = {e["name"]: e for e in (spec.get("home") or {}).get("entities", [])}

    def spoken(name: str) -> bool:
        phrases = [name, *(entities.get(name) or {}).get("aliases", [])]
        return any(
            len(p.split()) >= 2 and re.search(r"(?<!\w)" + re.escape(p.casefold()) + r"(?!\w)", lowered)
            for p in phrases
        )

    return sum(spoken(name) for name in candidates) == 1


def validate_utterance(spec: dict[str, Any]) -> str | None:
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        return "missing_utterance"
    if _BANNED.search(utterance):
        return "banned_marker"
    expected = spec.get("expected") or {}
    lowered = utterance.casefold()
    if _names_one_clarify_candidate(spec, lowered):
        return "clarify_target_named"
    if spec.get("area_scenario") or spec.get("category") == "area_grounding":
        for name in spec.get("excluded_names") or []:
            if "leave" not in lowered or name.casefold() not in lowered:
                return "missing_exclusion"
        return None
    if expected.get("kind") == "action":
        calls = expected.get("calls") or []
        for call in calls:
            arguments = call.get("arguments") or {}
            for scope in ("area", "floor"):
                if arguments.get(scope) and str(arguments[scope]).casefold() not in lowered:
                    return "missing_expected_scope"
            for key, value in arguments.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key in {"hours", "minutes", "seconds"} and value == 0:
                    continue
                if key == "volume_step":
                    value = abs(value)
                spellings = [str(value)]
                if float(value).is_integer():
                    spellings.append(str(int(value)))
                    words = _int_to_words(int(value))
                    if words:
                        spellings.append(words)
                if not any(
                    re.search(r"(?<![\w.])" + re.escape(word) + r"(?!\w|\.\d)", lowered)
                    for word in spellings
                ):
                    return "missing_expected_value"
        has_area_target = any(
            isinstance(c.get("arguments"), dict) and c["arguments"].get("area") and not c["arguments"].get("name")
            for c in calls
        )
        if (
            calls
            and not has_area_target
            and calls[0]["name"]
            not in {
                "HassCancelAllTimers",
                "HassStartTimer",
                "HassPauseTimer",
                "HassTimerStatus",
            }
        ):
            for name in spec.get("target_names") or []:
                if _is_satellite_name(name):
                    continue
                spoken = spec.get("spoken_targets", {}).get(name, name).casefold()
                if (
                    spoken not in lowered
                    and not utterance_contains_target(lowered, name)
                    and not utterance_contains_target(lowered, spoken)
                ):
                    area = (calls[0].get("arguments") or {}).get("area")
                    if not area or str(area).casefold() not in lowered:
                        if spec.get("discrimination"):
                            continue
                        return "missing_expected_target"
        excluded = spec.get("excluded_names") or []
        if not satellite_contrast_spoken(utterance, [
            str((call.get("arguments") or {}).get("name") or "") for call in calls
        ], excluded) and any("leave" not in lowered or name.casefold() not in lowered for name in excluded):
            return "missing_exclusion"
    if expected.get("kind") == "status" and not _STATUS_QUERY.search(lowered):
        return "status_not_query"
    return None


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _call_with_parsed_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return call
    return {**call, "function": {**function, "arguments": arguments}}


def count_row_tokens(row: dict[str, Any], *, model_name: str) -> int:
    tokenizer = _tokenizer(model_name)
    tools = []
    for tool in row.get("tools") or []:
        fn = tool.get("function", tool)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            }
        )
    messages = []
    for message in row.get("messages") or []:
        entry = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            entry["tool_calls"] = [_call_with_parsed_arguments(c) for c in message["tool_calls"]]
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        messages.append(entry)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def row_characters(row: dict[str, Any]) -> int:
    total = len(json.dumps(row.get("tools") or []))
    for message in row.get("messages") or []:
        total += len(str(message.get("content") or ""))
        total += len(json.dumps(message.get("tool_calls") or []))
    return total


_MIN_CHARS_PER_TOKEN = 3.0


def validate_token_budget(
    row: dict[str, Any],
    budget: int,
    *,
    tokenizer_model: str,
) -> str | None:
    if row_characters(row) / _MIN_CHARS_PER_TOKEN <= budget:
        return None
    estimated = False
    try:
        tokens = count_row_tokens(row, model_name=tokenizer_model)
    except Exception:
        from generators.context import serialize_context

        spec_len = len(serialize_context(row.get("metadata", {}))) + sum(
            len(str(m.get("content", ""))) for m in row.get("messages", [])
        )
        tokens = spec_len // 4
        estimated = True
    row.setdefault("metadata", {})["_token_length"] = tokens
    row["metadata"]["_token_length_estimated"] = estimated
    if tokens > budget:
        return "token_budget_exceeded"
    return None


def validate_row(
    spec: dict[str, Any],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    tokenizer_model: str = "LiquidAI/LFM2.5-230M-Base",
    rendered: dict[str, Any] | None = None,
) -> str | None:
    reason = validate_spec(spec)
    if reason:
        return reason
    reason = validate_utterance(spec)
    if reason:
        return reason
    if rendered is not None:
        return validate_token_budget(rendered, token_budget, tokenizer_model=tokenizer_model)
    return None


def validate_serialized_row(
    row: dict[str, Any],
    *,
    token_budget: int,
    tokenizer_model: str,
) -> str | None:
    if _PLACEHOLDER_AREA in json.dumps(row.get("messages") or []).casefold():
        return "placeholder_area"
    if not row.get("tools"):
        return "missing_tools"
    reason = (
        gate_offered_tools(row)
        or gate_scripts(row)
        or gate_areas(row)
        or gate_named_device_calls(row)
        or gate_clarification(row)
        or gate_on_off_polarity(row)
        or gate_hidden_capability_refusal(row)
    )
    if reason:
        return reason
    for message in row.get("messages", []):
        if message.get("role") == "assistant" and message.get("train_on_turn"):
            if message.get("tool_calls"):
                for call in message["tool_calls"]:
                    args = call.get("function", {}).get("arguments")
                    if isinstance(args, str) and len(args) > 4096:
                        return "truncated_supervision"
    return validate_token_budget(row, token_budget, tokenizer_model=tokenizer_model)


# Row-level checks for counterfactual groups (plan Step 5).

DEFAULT_JACCARD_THRESHOLD = 0.8
_BARE_TOOL = {namespaced_tool_name(tool): tool for tool in HA_TOOL_NAMESPACES}


def _system_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if messages and messages[0].get("role") == "system":
        return str(messages[0].get("content") or "")
    return ""


def _offered_schemas(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return tool_schema_map(row.get("tools") or [])


def _parsed_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@lru_cache(maxsize=512)
def _static_entities(system_prompt: str) -> tuple[dict[str, Any], ...]:
    """Entities listed in the static context: names (aliases included), domain, areas."""
    _, header, rest = system_prompt.partition(STATIC_CONTEXT_HEADER)
    if not header:
        return ()
    block: list[str] = []
    for line in rest.split("\n"):
        if line.startswith(("- ", " ")):
            block.append(line)
        elif block:
            break
    entities = []
    for entry in yaml.safe_load("\n".join(block)) or []:
        names = [name.strip() for name in str(entry.get("names") or "").split(", ") if name.strip()]
        areas = [area.strip() for area in str(entry.get("areas") or "").split(", ") if area.strip()]
        entities.append({"names": names, "domain": entry.get("domain"), "areas": areas})
    return tuple(entities)


@lru_cache(maxsize=None)
def _domain_supports(domain: str, tool: str) -> bool | None:
    """True/False when the domain has registry capabilities; None when it has none (not judged)."""
    caps = [cap for cap in CAPABILITIES.values() if cap.domain == domain]
    if not caps:
        return None
    return any(
        op.tool_name == tool and op.support != SupportLevel.UNAVAILABLE
        for cap in caps
        for op in cap.operations
    )


def check_unsupported_calls(row: dict[str, Any], *, home: dict[str, Any] | None = None) -> str | None:
    """Reason the row's assistant tool calls are not servable, or None.

    The row carries no entity features, so support is judged per domain unless
    `home` is given; with `home`, a named entity found there is checked with
    `_entity_can_run` (features included).
    """
    schemas = _offered_schemas(row)
    domains_by_name: dict[str, set] = defaultdict(set)
    for entity in _static_entities(_system_prompt(row)):
        for name in entity["names"]:
            domains_by_name[name].add(entity["domain"])
    home_by_name = {e["name"]: e for e in (home or {}).get("entities", [])}
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if name not in schemas:
                return "unoffered_tool"
            arguments = _parsed_arguments(function.get("arguments"))
            if arguments is None:
                return "invalid_call_shape"
            reason = validate_tool_arguments(name, arguments, schemas)
            if reason:
                return reason
            bare = _BARE_TOOL.get(name, name)
            target = arguments.get("name")
            if target is None or bare in _TIMER_NAME_TOOLS:
                continue
            if not isinstance(target, str):
                return "invalid_call_shape"
            domains = domains_by_name.get(target)
            if not domains:
                return "unknown_canonical_entity"
            if home is not None and target in home_by_name:
                if not _entity_can_run(home_by_name[target], bare):
                    return "entity_lacks_capability"
            elif all(_domain_supports(domain, bare) is False for domain in domains):
                return "unsupported_operation"
    return None


_KNOWN_HA_TOOLS = frozenset({*_BARE_TOOL, *HA_TOOL_NAMESPACES})
_AREA_CONTEXT_LINE = re.compile(r"^(?:target_area|satellite_area): (.+)$", re.MULTILINE)


def _labelled_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        call
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and message.get("train_on_turn")
        for call in message.get("tool_calls") or []
    ]


def gate_offered_tools(row: dict[str, Any]) -> str | None:
    offered = _offered_schemas(row)
    if any((call.get("function") or {}).get("name") not in offered for call in _labelled_calls(row)):
        return "unoffered_tool"
    return None


def gate_named_device_calls(row: dict[str, Any]) -> str | None:
    reason = check_unsupported_calls(row)
    if reason:
        return reason
    canonical = {entry["names"][0] for entry in _static_entities(_system_prompt(row)) if entry["names"]}
    calls = [
        tool_call
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls") or []
    ]
    for call in calls:
        function = call["function"]
        bare = _BARE_TOOL.get(function["name"], function["name"])
        target = (_parsed_arguments(function.get("arguments")) or {}).get("name")
        if bare in _KNOWN_HA_TOOLS and bare not in _TIMER_NAME_TOOLS and target is not None:
            if target not in canonical:
                return "unknown_canonical_entity"
    return None


def gate_scripts(row: dict[str, Any]) -> str | None:
    offered = _offered_schemas(row)
    for call in _labelled_calls(row):
        function = call.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or name in _KNOWN_HA_TOOLS or "__" in name:
            continue
        if name not in offered:
            return "unexposed_script"
        arguments = _parsed_arguments(function.get("arguments"))
        if arguments is None:
            return "invalid_call_shape"
        if arguments:
            return "script_tool_takes_no_arguments"
    return None


def _visible_areas(system_prompt: str) -> set[str]:
    areas = {area.casefold() for entry in _static_entities(system_prompt) for area in entry["areas"]}
    areas.update(value.casefold() for value in _AREA_CONTEXT_LINE.findall(system_prompt) if value != "none")
    return areas


def gate_areas(row: dict[str, Any]) -> str | None:
    visible = _visible_areas(_system_prompt(row))
    for call in _labelled_calls(row):
        arguments = _parsed_arguments((call.get("function") or {}).get("arguments"))
        if arguments is None:
            return "invalid_call_shape"
        area = arguments.get("area")
        if area is None:
            continue
        if not isinstance(area, str):
            return "invalid_call_shape"
        if area.casefold() not in visible:
            return "unknown_area"
    return None


_DID_YOU_MEAN = re.compile(r"Did you mean (.+)\?")
_WHICH_DEVICE = re.compile(r"Which (?:device|routine) did you mean\?")
_WHICH_IN_AREA = re.compile(r"Which (.+) do you mean, the one in the (.+) or the (.+)\?")
_CLARIFY_LIST_SEPARATOR = re.compile(r",\s*or\s+|,\s+|\s+or\s+")


def _clarify_candidates(text: str) -> list[tuple[str, str | None]] | None:
    if _WHICH_DEVICE.fullmatch(text):
        return []
    if match := _WHICH_IN_AREA.fullmatch(text):
        noun, first, second = match.groups()
        return [(noun, first), (noun, second)]
    if match := _DID_YOU_MEAN.fullmatch(text):
        return [(part, None) for part in _CLARIFY_LIST_SEPARATOR.split(match.group(1))]
    return None


def _spellings(text: str) -> set[str]:
    lowered = text.casefold()
    return {lowered, lowered.removeprefix("the ")}


def _clarification_entities(row: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    entries = list(_static_entities(_system_prompt(row)))
    for tool in row.get("tools") or []:
        schema = tool.get("function") or {}
        name = schema.get("name", "")
        if name in _KNOWN_HA_TOOLS or "__" in name:
            continue
        canonical, separator, aliases = schema.get("description", "").partition(". Aliases: ")
        names = [canonical]
        if separator:
            try:
                parsed = ast.literal_eval(aliases)
            except (SyntaxError, ValueError):
                parsed = []
            if isinstance(parsed, list):
                names.extend(alias for alias in parsed if isinstance(alias, str))
        entries.append({"names": names, "domain": "script", "areas": []})
    timers = set()
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if _BARE_TOOL.get(function.get("name"), function.get("name")) != "HassStartTimer":
                continue
            name = (_parsed_arguments(function.get("arguments")) or {}).get("name")
            if isinstance(name, str):
                timers.add(f"{name} timer".casefold())
    entries.extend({"names": [name], "domain": "timer", "areas": []} for name in sorted(timers))
    return tuple(entries)


def _resolve_candidate(text: str, area: str | None, entries: tuple[dict[str, Any], ...]) -> int | None:
    spellings = _spellings(text)
    matches = [
        index
        for index, entry in enumerate(entries)
        if spellings & {name.casefold() for name in entry["names"]}
        and (area is None or area.casefold() in {a.casefold() for a in entry["areas"]})
    ]
    return matches[0] if len(matches) == 1 else None


_CLARIFY_TYPE_WORDS = {
    cap.domain: {
        _type_label(cap.name),
        _type_label(cap.name).removesuffix("s"),
    }
    for cap in CAPABILITIES.values()
}
for _domain, _words in {
    "light": ("lamp",),
    "switch": ("plug",),
    "media_player": ("speaker", "tv", "television"),
    "script": ("routine",),
}.items():
    _CLARIFY_TYPE_WORDS[_domain].update(word for noun in _words for word in (noun, noun + "s"))
_CLARIFY_TYPES = {
    domain: re.compile(r"\b(?:" + "|".join(re.escape(word) for word in sorted(words)) + r")\b")
    for domain, words in _CLARIFY_TYPE_WORDS.items()
}
_TV_WORD = re.compile(r"\b(?:t[ .]*v|television)s?\b")


def _genuine_match(entry: dict[str, Any], domains: set[str], named_areas: set[str], tv: bool) -> bool:
    if entry["domain"] not in domains:
        return False
    if named_areas and entry["domain"] not in {"script", "timer"}:
        if not named_areas & {area.casefold() for area in entry["areas"]}:
            return False
    entity = {"name": entry["names"][0], "aliases": entry["names"][1:]}
    return not tv or any(refers_to(entity, word) for word in ("tv", "television"))


def _clarification_reason(candidates: list[tuple[str, str | None]], reference: str, row: dict[str, Any]) -> str | None:
    entries = _clarification_entities(row)
    identities = []
    for text, area in candidates:
        identity = _resolve_candidate(text, area, entries)
        if identity is None:
            return "clarify_unexposed_candidate"
        identities.append(identity)
    if len(set(identities)) < len(identities):
        return "clarify_duplicate_candidate"
    reference = reference.casefold()
    tv = bool(_TV_WORD.search(reference))
    domains = {"media_player"} if tv else {
        domain for domain, pattern in _CLARIFY_TYPES.items() if pattern.search(reference)
    }
    if not domains:
        return None
    areas = {
        area for area in _visible_areas(_system_prompt(row))
        if re.search(rf"\b{re.escape(area)}\b", reference)
    }
    if sum(_genuine_match(entries[index], domains, areas, tv) for index in identities) < 2:
        return "clarify_without_genuine_choice"
    return None


def gate_clarification(row: dict[str, Any]) -> str | None:
    messages = row.get("messages") or []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("train_on_turn") or message.get("tool_calls"):
            continue
        candidates = _clarify_candidates(str(message.get("content") or ""))
        if candidates is None:
            continue
        reference = next(
            (str(m.get("content") or "") for m in reversed(messages[:index]) if m.get("role") == "user"),
            "",
        )
        reason = _clarification_reason(candidates, reference, {**row, "messages": messages[:index]})
        if reason:
            return reason
    return None


_OHF_RULES = json.loads(Path(__file__).with_name("ohf_en.json").read_text())["rules"]
_SCENE_OPERATIONS = frozenset({"activate", "run"})


def _ohf_verbs(rule: str) -> frozenset[str]:
    group = re.search(r"\(([^)]*)\)", _OHF_RULES[rule])
    return frozenset(group.group(1).split("|")) if group else frozenset()


_ON_VERBS = (
    frozenset(op for op, tool in _OPERATION_TOOLS.items() if tool == "HassTurnOn" and "_" not in op) - _SCENE_OPERATIONS
) | _ohf_verbs("open")
_OFF_VERBS = frozenset(op for op, tool in _OPERATION_TOOLS.items() if tool == "HassTurnOff" and "_" not in op) | _ohf_verbs("close")
_TURN_VERBS = _ohf_verbs("turn")
_CLAUSE_BREAKS = frozenset({".", ",", ";", "!", "?", "and", "then", "but", "or"})


def _area_prepositions(tokens: list[str], places: set[str]) -> set[int]:
    phrases = {tuple(re.findall(r"[a-z]+", place)) for place in places} - {()}
    found = set()
    for index, token in enumerate(tokens):
        if token != "on":
            continue
        start = index + 1 + (tokens[index + 1:index + 2] == ["the"])
        for phrase in phrases:
            end = start + len(phrase)
            if tuple(tokens[start:end]) == phrase and tokens[end:end + 1] == ["off"]:
                found.add(index)
    return found


def _clause_words_after(tokens: list[str], start: int, prepositions: set[int]) -> list[str]:
    words = []
    for offset in range(start, len(tokens)):
        if tokens[offset] in _CLAUSE_BREAKS or tokens[offset] in _TURN_VERBS:
            break
        if offset not in prepositions:
            words.append(tokens[offset])
    return words


def _utterance_polarity(utterance: str, places: set[str]) -> str | None:
    tokens = re.findall(r"[a-z]+|[.,;!?]", utterance.casefold())
    prepositions = _area_prepositions(tokens, places)
    signals = {"off"} if prepositions else set()
    for index, token in enumerate(tokens):
        if index in prepositions:
            continue
        if token in _ON_VERBS:
            signals.add("on")
        if token in _OFF_VERBS:
            signals.add("off")
        if token in _TURN_VERBS:
            following = _clause_words_after(tokens, index + 1, prepositions)
            if "on" in following:
                signals.add("on")
            if "off" in following:
                signals.add("off")
    return signals.pop() if len(signals) == 1 else None


def _is_scene_or_script(target: str, system: str) -> bool:
    return any(
        entry["domain"] in {"scene", "script"} and target in entry["names"]
        for entry in _static_entities(system)
    )


def gate_on_off_polarity(row: dict[str, Any]) -> str | None:
    system = _system_prompt(row)
    visible = _visible_areas(system)
    messages = row.get("messages") or []
    utterance = ""
    for message in messages:
        if message.get("role") == "user":
            utterance = str(message.get("content") or "")
            continue
        if message.get("role") != "assistant" or not message.get("train_on_turn"):
            continue
        calls = message.get("tool_calls") or []
        bares = {_BARE_TOOL.get(name, name) for name in ((call.get("function") or {}).get("name") for call in calls)}
        if {"HassTurnOn", "HassTurnOff"} <= bares:
            continue
        floors = set()
        for call in calls:
            floor = (_parsed_arguments((call.get("function") or {}).get("arguments")) or {}).get("floor")
            if isinstance(floor, str):
                floors.add(floor.casefold())
        polarity = _utterance_polarity(utterance, visible | floors)
        if polarity is None:
            continue
        for call in calls:
            function = call.get("function") or {}
            bare = _BARE_TOOL.get(function.get("name"), function.get("name"))
            target = (_parsed_arguments(function.get("arguments")) or {}).get("name")
            if isinstance(target, str):
                if _is_scene_or_script(target, system) or {"on", "off"} & set(re.findall(r"[a-z]+", target.casefold())):
                    continue
            if bare == "HassTurnOn" and polarity == "off":
                return "turn_on_answers_off_utterance"
            if bare == "HassTurnOff" and polarity == "on":
                return "turn_off_answers_on_utterance"
    return None


_NAMED_REFUSALS = (
    re.compile(r"(.+) does not support that\."),
    re.compile(r"I can't control (.+)\. It is only available as context\."),
)
_WORD_TOOLS: dict[str, str] = {
    **{op: tool for op, tool in _OPERATION_TOOLS.items() if "_" not in op and op not in _SCENE_OPERATIONS},
    **{word: "HassTurnOn" for word in _ON_VERBS},
    **{word: "HassTurnOff" for word in _OFF_VERBS},
}


def _requested_tool(utterance: str) -> str | None:
    tokens = re.findall(r"[a-z]+", utterance.casefold())
    tools = {_WORD_TOOLS[token] for token in tokens if token in _WORD_TOOLS}
    for index, token in enumerate(tokens):
        if token in _TURN_VERBS:
            following = tokens[index + 1:index + 5]
            if "on" in following:
                tools.add("HassTurnOn")
            if "off" in following:
                tools.add("HassTurnOff")
    return next(iter(tools)) if len(tools) == 1 else None


def _named_refusal(text: str) -> str | None:
    for pattern in _NAMED_REFUSALS:
        if match := pattern.fullmatch(text):
            return match.group(1)
    return None


def gate_hidden_capability_refusal(row: dict[str, Any]) -> str | None:
    domains_by_name: dict[str, set[str]] = defaultdict(set)
    for entity in _static_entities(_system_prompt(row)):
        for name in entity["names"]:
            domains_by_name[name].add(entity["domain"])
    offered = _offered_schemas(row)
    utterance = ""
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            utterance = str(message.get("content") or "")
            continue
        if message.get("role") != "assistant" or not message.get("train_on_turn"):
            continue
        name = _named_refusal(str(message.get("content") or ""))
        domains = domains_by_name.get(name, set()) if name else set()
        tool = _requested_tool(utterance)
        if len(domains) != 1 or tool is None:
            continue
        if _domain_supports(next(iter(domains)), tool) is True and namespaced_tool_name(tool) in offered:
            return "hidden_capability_refusal"
    return None


def _label(row: dict[str, Any]) -> str:
    calls = sorted(
        (call["function"]["name"], json.dumps(_parsed_arguments(call["function"].get("arguments")), sort_keys=True))
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and message.get("train_on_turn")
        for call in message.get("tool_calls") or []
    )
    if calls:
        return json.dumps({"calls": calls})
    return json.dumps({"no_call": (row.get("metadata") or {}).get("no_action_reason")})


def _context_hash(row: dict[str, Any]) -> str:
    tools = sorted(tool["function"]["name"] for tool in row.get("tools") or [])
    return hashlib.sha256("\0".join([_system_prompt(row), *tools]).encode()).hexdigest()


def _command(row: dict[str, Any]) -> str:
    return "\n".join(
        _normalize_prompt(str(message.get("content") or ""))
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )


def check_label_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups that share a normalized command and model-visible context but disagree on the label."""
    groups: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        groups[(_command(row), _context_hash(row))][_label(row)].append(index)
    return [
        {"command": command, "labels": dict(labels)}
        for (command, _), labels in groups.items()
        if len(labels) > 1
    ]


@lru_cache(maxsize=1)
def _eval_index() -> dict[str, Any]:
    prompts = sorted(quality_eval_prompts())
    token_sets = [frozenset(prompt.split()) for prompt in prompts]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            postings[token].append(index)
    return {
        "sets": token_sets,
        "df": Counter({token: len(ids) for token, ids in postings.items()}),
        "postings": postings,
    }


def _jaccard_overlap(tokens: frozenset[str], index: dict[str, Any], threshold: float) -> bool:
    # Prefix filter: a pair with Jaccard >= t shares a token among the first
    # |U| - ceil(t|U|) + 1 tokens of U, ordered rarest-first in the eval set.
    ordered = sorted(tokens, key=lambda token: (index["df"].get(token, 0), token))
    prefix = len(ordered) - math.ceil(threshold * len(ordered)) + 1
    candidates: set[int] = set()
    for token in ordered[:prefix]:
        candidates.update(index["postings"].get(token, ()))
    return any(
        len(tokens & index["sets"][i]) / len(tokens | index["sets"][i]) >= threshold
        for i in candidates
    )


def check_fuzzy_eval_overlap(
    utterance: str,
    *,
    context: str = "",
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> bool:
    """Eval leakage: the utterance matches an eval prompt after normalization, or has token-set Jaccard >= threshold with one.

    `context` (the row's system prompt) is accepted for existing callers but not used for
    rejection. Masking entity names by context made generic shapes such as "turn off the <E>"
    match eval templates and rejected ordinary commands.
    """
    tokens = frozenset(_normalize_prompt(utterance).split())
    if not tokens:
        return False
    return _jaccard_overlap(tokens, _eval_index(), threshold)
