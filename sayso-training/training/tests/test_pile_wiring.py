from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from generators.config import GeneratorConfig
from generators.utterances import expand_utterance


def _entry(index: int, slot: str, template: str) -> dict:
    return {
        "id": f"wiring-{index}",
        "op": "lights.turn_on",
        "style": "direct",
        "template": template,
        "slot": slot,
        "writer": "writer-model",
        "verify": {"model": "verifier", "op": "lights.turn_on", "agree": True},
    }


def _write_pile(tmp_path: Path) -> tuple[Path, str]:
    entries = [
        _entry(0, "name", "go on and switch on the {name}"),
        _entry(1, "area", "fire up all the lights in the {area}"),
    ]
    path = tmp_path / "wording.jsonl"
    data = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries).encode("utf-8")
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _named_spec() -> dict:
    return {
        "capability": "lights",
        "operation": "turn_on",
        "seed": 7,
        "candidate_id": "wiring-named-0001",
        "category": "clean_direct",
        "home": {"entities": [{"name": "Kitchen Counter Light", "domain": "light", "area": "kitchen"}]},
        "expected": {
            "kind": "action",
            "calls": [{"name": "HassTurnOn", "arguments": {"name": "Kitchen Counter Light", "domain": ["light"]}}],
        },
        "target_names": ["Kitchen Counter Light"],
        "spoken_targets": {},
    }


def _area_spec() -> dict:
    return {
        "capability": "lights",
        "operation": "turn_on",
        "seed": 7,
        "candidate_id": "wiring-area-0001",
        "category": "clean_direct",
        "home": {"entities": [{"name": "Kitchen Counter Light", "domain": "light", "area": "kitchen"}]},
        "expected": {
            "kind": "action",
            "calls": [{"name": "HassTurnOn", "arguments": {"area": "kitchen", "domain": ["light"]}}],
        },
        "target_names": [],
        "spoken_targets": {},
    }


def test_disabled_config_keeps_base_utterance(tmp_path: Path):
    path, _ = _write_pile(tmp_path)
    GeneratorConfig()
    named = expand_utterance(_named_spec())
    area = expand_utterance(_area_spec())
    assert "switch on the" not in named and "fire up all the lights" not in named
    assert "fire up all the lights" not in area
    assert named == expand_utterance(_named_spec())


def test_enabled_share_one_replaces_with_pile_template(tmp_path: Path):
    path, sha = _write_pile(tmp_path)
    GeneratorConfig(wording_pile={"path": path, "sha256": sha, "share": 1.0})
    assert expand_utterance(_named_spec()) == "Go on and switch on the Kitchen Counter Light"


def test_area_slot_used_only_for_area_wide_calls(tmp_path: Path):
    path, sha = _write_pile(tmp_path)
    GeneratorConfig(wording_pile={"path": path, "sha256": sha, "share": 1.0})
    named = expand_utterance(_named_spec())
    area = expand_utterance(_area_spec())
    assert named == "Go on and switch on the Kitchen Counter Light"
    assert area == "Fire up all the lights in the kitchen"


def test_wrong_sha_raises(tmp_path: Path):
    path, _ = _write_pile(tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        GeneratorConfig(wording_pile={"path": path, "sha256": "0" * 64, "share": 0.3})
