from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from generators.piles import WordingPile, load_pile


def _entry(index: int, op: str = "lights.turn_on", slot: str = "name", template: str | None = None, agree: bool | None = True):
    entry = {
        "id": f"tpl-{index}",
        "op": op,
        "style": "direct",
        "template": template if template is not None else f"{op.replace('.', ' ')} phrasing number {index} for {{name}}",
        "slot": slot,
        "writer": "writer-model",
    }
    if agree is not None:
        entry["verify"] = {"model": "verifier", "op": op, "agree": agree}
    return entry


def _write_pile(tmp_path: Path, entries: list[dict]) -> tuple[Path, str]:
    path = tmp_path / "pile.jsonl"
    data = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries).encode("utf-8")
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.fixture()
def pile_file(tmp_path: Path):
    entries = [
        _entry(0, "lights.turn_on", "name"),
        _entry(1, "lights.turn_on", "name"),
        _entry(2, "lights.turn_on", "area", template="switch the lights on in the {area}"),
        _entry(3, "lights.turn_off", "name"),
        _entry(4, "lights.set_brightness", "name", agree=False),
        _entry(5, "lights.set_brightness", "name", agree=None),
    ]
    return _write_pile(tmp_path, entries)


def test_sha_mismatch_raises_value_error(pile_file):
    path, sha = pile_file
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_pile(path, "0" * 64)


def test_keeps_only_agreed_entries_in_file_order(pile_file):
    path, sha = pile_file
    pile = load_pile(path, sha)
    assert [entry["id"] for entry in pile.entries] == ["tpl-0", "tpl-1", "tpl-2", "tpl-3"]
    assert pile.ops() == ["lights.turn_off", "lights.turn_on"]


def test_load_is_cached_by_path_and_sha(pile_file):
    path, sha = pile_file
    assert load_pile(path, sha) is load_pile(path, sha)
    assert load_pile(str(path), sha) is load_pile(Path(str(path)), sha.upper())


def test_choose_is_deterministic_for_a_fixed_seed(pile_file):
    path, sha = pile_file
    pile = load_pile(path, sha)
    draws = []
    for _ in range(2):
        rng = random.Random(1234)
        draws.append([pile.choose("lights.turn_on", "name", rng)["id"] for _ in range(50)])
    assert draws[0] == draws[1]
    assert all(entry_id in ("tpl-0", "tpl-1") for entry_id in draws[0])


def test_render_substitutes_name_and_area_slots(pile_file):
    path, sha = pile_file
    pile = load_pile(path, sha)
    name_entry = next(entry for entry in pile.entries if entry["slot"] == "name")
    area_entry = next(entry for entry in pile.entries if entry["slot"] == "area")
    assert pile.render(name_entry, "Kitchen Counter Light")
    assert pile.render(area_entry, "the kitchen") == "switch the lights on in the the kitchen"


def test_render_trims_surrounding_whitespace():
    entry = _entry(9, template="  turn on {name}  ")
    pile = WordingPile([entry])
    assert pile.render(entry, "Lamp") == "turn on Lamp"


def test_choose_returns_none_for_uncovered_op(pile_file):
    path, sha = pile_file
    pile = load_pile(path, sha)
    rng = random.Random(0)
    assert pile.choose("vacuum.start", "name", rng) is None
    assert pile.choose("lights.set_brightness", "name", rng) is None
