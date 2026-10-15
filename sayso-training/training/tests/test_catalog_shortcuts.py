
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.homes import _capability_slots, generate_home
from generators.rendering import _decoy_removals
from generators.stt_noise import apply_log_stt_noise
from generators.utterances import request_seed_from_spec
from generators.validation import _STATUS_QUERY


def test_script_count_is_not_size_scaled():
    for size in (16, 64):
        counts = {_capability_slots(size, random.Random(seed)).count("scripts") for seed in range(200)}
        assert {1, 2, 3, 4} <= counts, (size, counts)


def test_decoys_withhold_only_unneeded_tools_at_the_set_rate():
    home = generate_home(0, 64, random.Random(1))
    messages = [{"role": "assistant", "tool_calls": [{"function": {"name": "intent__HassTurnOn"}}]}]
    withheld = 0
    for i in range(1000):
        spec = {
            "candidate_id": f"c{i}",
            "home": home,
            "capability": "lights",
            "operation": "set_brightness",
            "expected": {"kind": "action", "calls": []},
        }
        removed = _decoy_removals(spec, messages)
        assert "intent__HassTurnOn" not in removed
        assert "light__HassLightSet" not in removed
        withheld += bool(removed)
    assert 0.25 < withheld / 1000 < 0.35


def test_status_seeds_include_yes_no_questions_that_validate():
    seeds = [
        request_seed_from_spec({
            "candidate_id": f"c{i}",
            "target_names": ["Front Door Lock"],
            "expected": {"kind": "status", "calls": [
                {"name": "GetLiveContext", "arguments": {"domain": "lock", "name": "Front Door Lock"}}
            ]},
        })
        for i in range(200)
    ]
    assert any(s.startswith("is ") for s in seeds)
    assert any("status of" in s for s in seeds)
    assert all(_STATUS_QUERY.search(s) for s in seeds)


def test_log_stt_period_never_follows_a_question_mark():
    for seed in range(300):
        text, _ = apply_log_stt_noise("Is the kitchen light on?", random.Random(seed))
        assert not text.rstrip().endswith("?."), text


def test_plan_reserves_the_tool_floor_for_every_covered_tool():
    import collections

    from generators.capability_registry import CAPABILITIES, trainable_operations
    from generators.config import GeneratorConfig
    from generators.coverage import expected_tool
    from generators.planning import _FLOOR_HEADROOM, build_plan

    cfg = GeneratorConfig.from_yaml(ROOT / "configs/generation/full_sft_v5.yaml", repo_root=ROOT.parent)
    plan = build_plan(cfg.count, cfg.seed, cfg.allocations, min_positive_per_tool=200)
    slots = collections.Counter(
        expected_tool(s.capability, s.operation) for s in plan.slots if s.family in {"ordinary", "settings"}
    )
    tools = {expected_tool(n, op.name) for n, cap in CAPABILITIES.items() for op in trainable_operations(cap)}
    for tool in tools - {None}:
        assert slots[tool] >= int(200 * _FLOOR_HEADROOM), (tool, slots[tool])


def test_lock_labels_carry_no_device_class():
    from generators.tools import build_area_call, build_turn_off, build_turn_on

    lock = {"name": "Front Door Lock", "domain": "lock", "device_class": "door", "capability": "locks"}
    assert build_turn_on(lock)["arguments"] == {"name": "Front Door Lock"}
    assert build_turn_off(lock)["arguments"] == {"name": "Front Door Lock"}
    area = build_area_call("locks", "lock", "Garage")["arguments"]
    assert area == {"area": "Garage", "domain": ["lock"]}


def test_single_entity_on_off_labels_keep_the_target_domain():
    import json

    from generators.rendering import render_example

    home = generate_home(0, 64, random.Random(1))
    light = next(e for e in home["entities"] if e["domain"] == "light")
    on_call = {"name": "HassTurnOn", "arguments": {"name": light["name"], "domain": ["light"]}}
    spec = {
        "candidate_id": "c-domain-strip",
        "home": home,
        "utterance": f"Turn on the {light['name']}",
        "target_names": [light["name"]],
        "expected": {"kind": "action", "calls": [on_call]},
    }
    row = render_example(spec)
    calls = next(m for m in row["messages"] if m.get("tool_calls"))["tool_calls"]
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"name": light["name"], "domain": ["light"]}
    assert spec["expected"]["calls"][0]["arguments"]["domain"] == ["light"]

    off_call = {"name": "HassTurnOff", "arguments": {"area": "Kitchen", "domain": ["light"]}}
    area_spec = {
        "candidate_id": "c-domain-keep",
        "home": home,
        "utterance": "Turn off the kitchen lights",
        "expected": {"kind": "action", "calls": [off_call]},
    }
    area_row = render_example(area_spec)
    area_calls = next(m for m in area_row["messages"] if m.get("tool_calls"))["tool_calls"]
    assert json.loads(area_calls[0]["function"]["arguments"]) == {"area": "Kitchen", "domain": ["light"]}



def test_name_targeted_on_off_labels_carry_their_entity_domain():
    import json

    from generators.config import GeneratorConfig
    from generators.pipeline import run_generation
    from generators.real_home import load_real_home

    fixture = ROOT / "fixtures" / "real_home.json"
    config = GeneratorConfig.from_yaml(ROOT / "configs/generation/smoke.yaml", repo_root=ROOT.parent)
    config.area_distribution_path = None
    config.count = 120
    config.seed = 73
    config.real_home_path = fixture
    config.real_home_rate = 1.0
    config.synthetic_only = False
    config.allocations = {"ordinary": 0.35, "status": 0.2, "settings": 0.15, "multi_action": 0.15, "correction": 0.15}
    home = load_real_home(fixture, split=config.split)
    entity_domain: dict[str, str] = {}
    for entity in home["entities"]:
        for label in [entity["name"], *entity.get("aliases", [])]:
            entity_domain[label.casefold()] = entity["domain"]
    context_only = {c["name"].casefold() for c in home.get("context_entities", [])} - set(entity_domain)

    rows = run_generation(config)["rows"]
    checked = 0
    for row in rows:
        if not row["metadata"].get("real_home"):
            continue
        for message in row["messages"]:
            for call in message.get("tool_calls") or []:
                tool_name = call["function"]["name"]
                if not tool_name.endswith(("HassTurnOn", "HassTurnOff")):
                    continue
                args = json.loads(call["function"]["arguments"])
                if "name" not in args or "area" in args or "floor" in args:
                    continue
                key = args["name"].casefold()
                assert key not in context_only, (tool_name, args)
                if key not in entity_domain:
                    continue
                checked += 1
                assert "domain" in args, (tool_name, args)
                assert all(d == entity_domain[key] for d in args["domain"]), (args, entity_domain[key])
    assert checked >= 10, checked
