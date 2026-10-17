
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_NORMALIZE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_NORMALIZE.sub(" ", text.casefold()).split())


def utterance_hash(utterance: str) -> str:
    return hashlib.sha256(_normalize(utterance).encode()).hexdigest()


def context_hash(home: dict[str, Any]) -> str:
    identity = home.get("home_id")
    if identity is None:
        payload = json.dumps(
            [{"name": e["name"], "area": e["area"]} for e in home.get("entities", [])],
            sort_keys=True,
        )
    else:
        payload = json.dumps({"home_id": identity}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def pair_hash(utterance: str, home: dict[str, Any]) -> str:
    return hashlib.sha256(f"{utterance_hash(utterance)}:{context_hash(home)}".encode()).hexdigest()


def input_hash(spec: dict[str, Any]) -> str:
    """Hash of the request, home identity, offered tools and prior turns.

    Keying on utterance and home alone let a sentence appear only once per home,
    whatever its tool catalog or conversation, which ruled out counterfactuals.
    """
    home = spec.get("home", {})
    expected = spec.get("expected") or {}
    payload = {
        "utterance": _normalize(spec.get("utterance") or ""),
        "home": context_hash(home),
        "removed_tools": sorted(expected.get("unavailable_tools") or spec.get("removed_tools") or []),
        "turns": [spec.get(key) for key in ("correction_wrong_name", "follow_up_reply", "follow_up_clarify_candidates")],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class DuplicateTracker:
    def __init__(self, near_limit: int = 3) -> None:
        self.near_limit = near_limit
        self._pair_counts: dict[str, int] = {}
        self._semantic_counts: dict[str, int] = {}

    def occurrences(self, spec: dict[str, Any]) -> int:
        return self._pair_counts.get(input_hash(spec), 0)

    def would_reject(self, spec: dict[str, Any]) -> str | None:
        ph = input_hash(spec)
        if ph in self._pair_counts:
            return "exact_duplicate_utterance"
        sem = spec.get("semantic_id") or spec.get("candidate_id")
        if self._semantic_counts.get(sem, 0) >= spec.get("_near_limit", self.near_limit):
            return "duplicate_semantic_id"
        return None

    def record(self, spec: dict[str, Any]) -> None:
        ph = input_hash(spec)
        self._pair_counts[ph] = self._pair_counts.get(ph, 0) + 1
        sem = spec.get("semantic_id") or spec.get("candidate_id")
        self._semantic_counts[sem] = self._semantic_counts.get(sem, 0) + 1

__all__ = [
    "DuplicateTracker",
    "context_hash",
    "input_hash",
    "pair_hash",
    "utterance_hash",
]
