from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

PileEntry = dict[str, Any]

_CACHE: dict[tuple[str, str], WordingPile] = {}


class WordingPile:
    def __init__(self, entries: list[PileEntry]):
        self.entries = list(entries)
        self._by_cell: dict[tuple[str, str], list[PileEntry]] = {}
        for entry in self.entries:
            self._by_cell.setdefault((entry["op"], entry["slot"]), []).append(entry)

    def choose(self, op: str, slot: str, rng: random.Random) -> PileEntry | None:
        options = self._by_cell.get((op, slot))
        if not options:
            return None
        return rng.choice(options)

    def render(self, entry: PileEntry, value: str) -> str:
        return entry["template"].replace("{" + entry["slot"] + "}", value).strip()

    def ops(self) -> list[str]:
        return sorted({entry["op"] for entry in self.entries})


def load_pile(path: Any, sha256: str) -> WordingPile:
    key = (os.fspath(path), sha256.strip().lower())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    with open(key[0], "rb") as handle:
        data = handle.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != key[1]:
        raise ValueError(f"pile sha256 mismatch for {key[0]}: expected {key[1]}, got {actual}")
    entries = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    kept = [entry for entry in entries if (entry.get("verify") or {}).get("agree") is True]
    pile = WordingPile(kept)
    _CACHE[key] = pile
    return pile
