#!/usr/bin/env python3
"""Build the slotted wording pile: writer phrasings, blind verifier agreement, sha-pinned JSONL.

Offline only. Generation does not import this module; the recipe pins the output by sha256.
Subcommands: write (writer model), verify (verifier model), finalize (agreed entries + manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for path in (ROOT, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals import cases as eval_cases  # noqa: E402
from generators.capability_registry import (  # noqa: E402
    CAPABILITIES,
    CapabilitySpec,
    OperationSpec,
    SupportLevel,
    trainable_operations,
)

STYLES: dict[str, str] = {
    "direct": "a direct imperative command",
    "polite": "a polite request, such as starting with 'please' or 'could you'",
    "indirect": "an indirect phrasing such as \"I'd like ...\", \"get ... running\", or \"I need ...\"",
    "terse": "very short and clipped",
    "casual": "relaxed everyday speech",
    "question": "a question asking the assistant to make it happen",
}
SLOTS: tuple[str, ...] = ("name", "area")
DEVICE_NOUNS = frozenset({
    "light", "lights", "lamp", "lamps", "fan", "fans", "tv", "plug", "outlet", "switch", "lock",
    "blind", "blinds", "shade", "vacuum", "speaker", "thermostat", "timer",
})
AREA_NOUNS: dict[str, frozenset[str]] = {
    "lights": frozenset({"light", "lights", "lighting", "lamps"}),
    "fans": frozenset({"fan", "fans"}),
    "covers": frozenset({"blinds", "shades", "curtains", "covers"}),
    "media_players": frozenset({"tv", "tvs", "speaker", "speakers", "music"}),
    "scenes": frozenset({"scene", "scenes"}),
    "switches": frozenset({"plugs", "outlets", "switches"}),
    "vacuums": frozenset({"vacuum", "vacuums"}),
    "climate": frozenset({"thermostat", "thermostats", "heating", "climate"}),
    "locks": frozenset({"lock", "locks", "doors"}),
}
DEFAULT_N = 40
DEFAULT_URL = "http://127.0.0.1:8000/v1"  # VM-local llama-server
MIN_WORDS, MAX_WORDS = 2, 14
MAX_TOKENS = 4096
ATTEMPTS = 3
VERIFY_BATCH = 20
CHECKPOINT_EVERY = 10
REVIEW_SAMPLE = 50
AMBIGUOUS = "ambiguous"
UNNATURAL = "unnatural"

_SLOT = re.compile(r"\{([^{}]*)\}")
_WORD = re.compile(r"\{(?:name|area)\}|[a-z]+")


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    api_key: str | None
    no_think: bool = False


def chat(endpoint: Endpoint, prompt: str, temperature: float, timeout: float = 600) -> str:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    body = {
        "model": endpoint.model,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if endpoint.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(
        endpoint.url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"] or ""


def parse_json_array(text: str) -> list[Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, list) else None


def ask(
    endpoint: Endpoint,
    prompt: str,
    temperature: float,
    accept: Callable[[list[Any]], bool],
    label: str,
) -> list[Any] | None:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            text = chat(endpoint, prompt, temperature)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as err:
            print(f"{label}: attempt {attempt} failed: {err}", file=sys.stderr)
        else:
            items = parse_json_array(text)
            if items is not None and accept(items):
                return items
            print(f"{label}: attempt {attempt} unparseable or wrong shape", file=sys.stderr)
        if attempt < ATTEMPTS:
            time.sleep(2**attempt)
    return None


def wording_operations() -> list[tuple[str, CapabilitySpec, OperationSpec]]:
    return [
        (f"{cap.name}.{op.name}", cap, op)
        for cap in CAPABILITIES.values()
        for op in trainable_operations(cap)
    ]


def select_operations(spec: str | None) -> list[tuple[str, CapabilitySpec, OperationSpec]]:
    ops = wording_operations()
    if not spec:
        return ops
    wanted = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = wanted - {key for key, _, _ in ops}
    if unknown:
        raise SystemExit(f"unknown operations: {', '.join(sorted(unknown))}")
    return [item for item in ops if item[0] in wanted]


def supports_area(cap: CapabilitySpec, op: OperationSpec) -> bool:
    return "area" in cap.targeting_modes and op.support is SupportLevel.SUPPORTED


def write_prompt(op_key: str, cap: CapabilitySpec, op: OperationSpec, style: str, n: int) -> str:
    slot_rule = "Each phrasing must contain exactly one slot: {name} for the named device"
    noun_rule = "Do not put a device noun such as light, fan or tv right after {name}. "
    if supports_area(cap, op):
        slot_rule += " or {area} for the whole area"
        noun_rule += "With {area}, also name the devices, for example 'turn on {area} lights'. "
    gloss = f"{cap.name.replace('_', ' ')}: {op.name.replace('_', ' ')}"
    return (
        f"Write {n} different things a person in a home might say to a voice assistant "
        f"to make this happen ({op_key}; {gloss}).\n"
        f"Style: {STYLES[style]}.\n"
        f"{slot_rule}. Do not use any other braces. {noun_rule}Each phrasing must be "
        f"{MIN_WORDS} to {MAX_WORDS} words. Vary wording and sentence shape.\n"
        f"Reply with only a JSON array of {n} strings, no commentary."
    )


def verify_options(cap_name: str) -> list[str]:
    return [key for key, cap, _ in wording_operations() if cap.name == cap_name] + [AMBIGUOUS, UNNATURAL]


def verify_prompt(templates: list[str], options: list[str], cap_name: str) -> str:
    slot_hint = (
        "In each, {name} stands for the timer's name (such as 'laundry'), not a device name.\n"
        if cap_name == "timers"
        else "In each, {name} stands for a device name and {area} stands for an area name.\n"
    )
    return (
        f"Below is a JSON array of {len(templates)} voice-assistant command templates. "
        + slot_hint
        + f"For each template, choose the one operation it asks for from: {', '.join(options)}.\n"
        f"Use \"{AMBIGUOUS}\" if it does not clearly ask for exactly one of them.\n"
        'Use "' + UNNATURAL + '" if a person at home would not plausibly say the template to a '
        "voice assistant, such as stilted or technical wording ('Trigger the fan action assigned to {name}', "
        "'Enable lighting via {name}') or wording that does not actually describe the operation.\n"
        f"Reply with only a JSON array of {len(templates)} strings, one choice per template, "
        "in the same order.\n"
        f"Templates: {json.dumps(templates, ensure_ascii=False)}"
    )


def slot_check(template: str, area_ok: bool) -> str | None:
    """Return the name of the filter that rejects this template, or None when it is valid."""
    slots = _SLOT.findall(template)
    if "{" in _SLOT.sub("", template) or "}" in _SLOT.sub("", template):
        return "braces"
    if len(slots) != 1 or slots[0] not in SLOTS:
        return "slot"
    if slots[0] == "area" and not area_ok:
        return "area_slot"
    if not MIN_WORDS <= len(template.split()) <= MAX_WORDS:
        return "words"
    return None


def noun_check(template: str, cap: CapabilitySpec) -> str | None:
    """Reject a {name} followed by a device noun, and an {area} with no noun for the capability."""
    words = _WORD.findall(template.casefold())
    for before, after in zip(words, words[1:]):
        if before != "{name}" or after not in DEVICE_NOUNS:
            continue
        if cap.name == "timers" and after == "timer":
            continue
        return "doubled_noun"
    if "{area}" in words and not AREA_NOUNS.get(cap.name, frozenset()) & set(words):
        return "area_noun"
    return None


def template_key(template: str) -> str:
    text = template.replace("{name}", " slotname ").replace("{area}", " slotarea ")
    return eval_cases.normalize_utterance(text)


def template_id(template: str) -> str:
    return hashlib.sha1(template_key(template).encode("utf-8")).hexdigest()


def eval_template(utterance: str, names: set[str], areas: set[str]) -> str:
    lowered_names = {name.casefold() for name in names}
    alternatives = sorted(names | areas, key=len, reverse=True)
    if not alternatives:
        return utterance
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(alt) for alt in alternatives) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern.sub(
        lambda match: "{name}" if match.group(1).casefold() in lowered_names else "{area}",
        utterance,
    )


def home_names(home: dict[str, Any]) -> tuple[set[str], set[str]]:
    areas: set[str] = set(home.get("areas") or {}) | set(home.get("area_aliases") or {})
    for alias_list in list((home.get("areas") or {}).values()) + list((home.get("area_aliases") or {}).values()):
        areas.update(alias_list or [])
    names: set[str] = set()
    for entity in home.get("entities") or []:
        names.add(entity["name"])
        names.update(entity.get("aliases") or [])
        if entity.get("area"):
            areas.add(entity["area"])
    return names, areas


def eval_keys() -> set[str]:
    homes: dict[str, dict[str, Any]] = {}
    keys: set[str] = set()
    for case in eval_cases.load_all_cases():
        if case.household not in homes:
            homes[case.household] = eval_cases.load_home(case.household)
        names, areas = home_names(homes[case.household])
        keys.add(template_key(eval_template(case.utterance, names, areas)))
    return keys


def admit(
    candidates: list[str],
    *,
    op_key: str,
    cap: CapabilitySpec,
    op: OperationSpec,
    style: str,
    writer: str,
    seen: set[str],
    banned: set[str],
    counts: Counter,
) -> list[dict[str, Any]]:
    accepted = []
    for raw in candidates:
        counts["candidates"] += 1
        template = " ".join(raw.split())
        reason = slot_check(template, supports_area(cap, op)) or noun_check(template, cap)
        if reason:
            counts[reason] += 1
            continue
        key = template_key(template)
        if key in seen:
            counts["duplicate"] += 1
            continue
        if key in banned:
            counts["eval_overlap"] += 1
            continue
        seen.add(key)
        accepted.append(
            {
                "id": template_id(template),
                "op": op_key,
                "style": style,
                "template": template,
                "slot": "area" if "{area}" in template else "name",
                "writer": writer,
            }
        )
    return accepted


def sort_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (entry["op"], entry["style"], entry["id"])


def pile_bytes(entries: list[dict[str, Any]]) -> bytes:
    lines = (json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in sorted(entries, key=sort_key))
    return "".join(line + "\n" for line in lines).encode("utf-8")


def read_pile(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def make_endpoint(args: argparse.Namespace) -> Endpoint:
    key = None
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "")
        if not key:
            raise SystemExit(f"{args.api_key_env} is unset or empty")
    return Endpoint(args.url, args.model, key, args.no_think)


def cmd_write(args: argparse.Namespace) -> int:
    endpoint = make_endpoint(args)
    banned = eval_keys()
    seen: set[str] = set()
    counts: Counter = Counter()
    entries: list[dict[str, Any]] = []
    for op_key, cap, op in select_operations(args.ops):
        for style in STYLES:
            prompt = write_prompt(op_key, cap, op, style, args.n)
            items = ask(endpoint, prompt, args.temperature, lambda items: bool(items), f"write {op_key}/{style}")
            if items is None:
                counts["cell_failed"] += 1
                continue
            strings = [item for item in items if isinstance(item, str)]
            counts["malformed"] += len(items) - len(strings)
            entries += admit(
                strings, op_key=op_key, cap=cap, op=op, style=style,
                writer=endpoint.model, seen=seen, banned=banned, counts=counts,
            )
    write_atomic(args.out, pile_bytes(entries))
    print(f"wrote {len(entries)} templates to {args.out}", file=sys.stderr)
    print("filters:", dict(sorted(counts.items())), file=sys.stderr)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    endpoint = make_endpoint(args)
    entries = read_pile(args.pile)
    by_cap: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        if entry.get("verify", {}).get("model") == endpoint.model:
            continue
        by_cap.setdefault(entry["op"].split(".")[0], []).append(index)
    batches = [
        (cap_name, indexes[start : start + VERIFY_BATCH])
        for cap_name, indexes in sorted(by_cap.items())
        for start in range(0, len(indexes), VERIFY_BATCH)
    ]
    counts: Counter = Counter()
    try:
        for number, (cap_name, chunk) in enumerate(batches, 1):
            templates = [entries[index]["template"] for index in chunk]
            prompt = verify_prompt(templates, verify_options(cap_name), cap_name)
            items = ask(
                endpoint, prompt, args.temperature,
                lambda items, size=len(chunk): len(items) == size,
                f"verify {cap_name} batch {number}",
            )
            if items is None:
                counts["failed"] += len(chunk)
                continue
            for index, item in zip(chunk, items):
                choice = item.strip() if isinstance(item, str) else ""
                agree = choice == entries[index]["op"]
                entries[index]["verify"] = {"model": endpoint.model, "op": choice, "agree": agree}
                if choice == UNNATURAL:
                    counts[UNNATURAL] += 1
                else:
                    counts["agree" if agree else "disagree"] += 1
            if number % CHECKPOINT_EVERY == 0:
                write_atomic(args.pile, pile_bytes(entries))
    finally:
        write_atomic(args.pile, pile_bytes(entries))
    print(f"verify: {dict(sorted(counts.items()))}", file=sys.stderr)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    entries = read_pile(args.pile)
    kept = [entry for entry in entries if entry.get("verify", {}).get("agree") is True]
    if not kept:
        raise SystemExit("no entries agreed with the verifier")
    writers = {entry["writer"] for entry in kept}
    verifiers = {entry["verify"]["model"] for entry in kept}
    if len(writers) != 1 or len(verifiers) != 1:
        raise SystemExit(f"pile mixes writers {sorted(writers)} or verifiers {sorted(verifiers)}")
    data = pile_bytes(kept)
    write_atomic(args.out, data)
    manifest = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "count": len(kept),
        "per_op": dict(sorted(Counter(entry["op"] for entry in kept).items())),
        "writer": writers.pop(),
        "verifier": verifiers.pop(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pile {args.out} sha256={manifest['sha256']} count={manifest['count']}")
    ordered = sorted(kept, key=sort_key)
    rng = random.Random(args.seed)
    for entry in rng.sample(ordered, min(REVIEW_SAMPLE, len(ordered))):
        print(f"{entry['op']}\t{entry['style']}\t{entry['template']}")
    return 0


def add_endpoint_args(parser: argparse.ArgumentParser, no_think: bool) -> None:
    parser.add_argument("--url", default=DEFAULT_URL, help=f"OpenAI-compatible base URL (default {DEFAULT_URL})")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", help="name of an env var holding the API key (optional)")
    parser.add_argument("--no-think", dest="no_think", action="store_true",
                        help="send chat_template_kwargs enable_thinking=false")
    parser.add_argument("--think", dest="no_think", action="store_false", help="allow model thinking")
    parser.set_defaults(no_think=no_think)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "examples:\n"
            "  build_wording_pile.py write --model QWEN --out pile.jsonl --ops lights.turn_on --n 5\n"
            "  build_wording_pile.py verify --model QWEN --pile pile.jsonl\n"
            "  build_wording_pile.py finalize --pile pile.jsonl --out pile.final.jsonl --manifest manifest.json\n"
            f"default --url is {DEFAULT_URL}; --api-key-env NAME reads the key from that env var"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="ask the writer for slotted phrasings per (op, style)")
    add_endpoint_args(write, no_think=False)
    write.add_argument("--out", type=Path, required=True)
    write.add_argument("--ops", help="comma-separated capability.operation names; default all")
    write.add_argument("--n", type=int, default=DEFAULT_N, help="phrasings per cell")
    write.add_argument("--temperature", type=float, default=0.9)
    write.set_defaults(func=cmd_write)

    verify = sub.add_parser("verify", help="blind-classify each template; resumable")
    add_endpoint_args(verify, no_think=True)
    verify.add_argument("--pile", type=Path, required=True, help="pile JSONL, updated in place")
    verify.add_argument("--temperature", type=float, default=0.0)
    verify.set_defaults(func=cmd_verify)

    finalize = sub.add_parser("finalize", help="keep agreed entries; write pile and manifest")
    finalize.add_argument("--pile", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--seed", type=int, default=1, help="seed for the review sample")
    finalize.set_defaults(func=cmd_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
