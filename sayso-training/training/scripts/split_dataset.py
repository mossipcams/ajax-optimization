#!/usr/bin/env python3
"""Split a JSONL dataset into train/val/test by whole scenario group.

Rows are linked through three keys: the contrast group, the scenario (home id
plus semantic id), and the household signature (the model-visible entity set).
Linked rows always land in the same split, so no synthetic household or contrast
group appears on both sides of a split.

The real home is the deployment home, not a synthetic household: its rows carry
no household key, so they split by scenario like any other group instead of
collapsing into one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_CONTEXT = "Static Context:"
REAL_HOME_PREFIXES = ("real_home", "synthetic_reference_home")


class _Links:
    """Union-find over group keys; each root is the smallest key in its component."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, first: str, second: str) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def _sorted_csv(value: str) -> str:
    return ", ".join(sorted(part.strip() for part in value.split(",") if part.strip()))


def _household_signature(record: dict) -> str | None:
    system = next(
        (m.get("content") or "" for m in record.get("messages") or [] if m.get("role") == "system"),
        "",
    )
    if STATIC_CONTEXT not in system:
        return None
    entities: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in system.split(STATIC_CONTEXT, 1)[1].splitlines():
        line = raw.strip()
        if not line:
            continue
        if raw.startswith("- names:"):
            current = {"names": line.removeprefix("- names:"), "domain": "", "areas": ""}
            entities.append(current)
        elif current is not None and raw[0].isspace():
            # Indented lines are either a domain/areas field or a wrapped names line.
            if line.startswith(("domain:", "areas:")):
                key, _, value = line.partition(":")
                current[key] = value.strip()
            else:
                current["names"] += " " + line
        elif entities:
            break
    if not entities:
        return None
    entity_set = sorted(
        (_sorted_csv(item["names"]), item["domain"], _sorted_csv(item["areas"])) for item in entities
    )
    return hashlib.sha256(json.dumps(entity_set).encode()).hexdigest()[:16]


def _legacy_key(record: dict, seed: int) -> str:
    meta = record.get("metadata") or {}
    template = meta.get("template_family") or meta.get("template") or ""
    phrasing = meta.get("phrasing_family") or meta.get("phrasing") or ""
    gen_seed = str(meta.get("seed", seed))
    if not template and not phrasing:
        messages = record.get("messages") or []
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        template = hashlib.sha256(str(user).encode()).hexdigest()[:8]
    return f"{template}|{phrasing}|{gen_seed}"


def _is_real_home(record: dict) -> bool:
    meta = record.get("metadata") or {}
    home_id = str(meta.get("home_id") or "")
    return bool(meta.get("real_home")) or home_id.startswith(REAL_HOME_PREFIXES)


def _group_keys(record: dict, seed: int) -> list[str]:
    meta = record.get("metadata") or {}
    keys = []
    if meta.get("contrast_group"):
        keys.append(f"contrast|{meta['contrast_group']}")
    if meta.get("semantic_id"):
        keys.append(f"scenario|{meta.get('home_id') or ''}|{meta['semantic_id']}")
    # The real home shares one entity set across all its rows; linking them by it
    # would pull the whole real home into a single split.
    household = None if _is_real_home(record) else _household_signature(record)
    if household:
        keys.append(f"household|{household}")
    if not keys:
        keys.append(f"legacy|{_legacy_key(record, seed)}")
    return keys


def _kind_key(key: str) -> str:
    return key.split("|", 1)[0]


def assign_split(family: str, *, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    digest = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 1000
    train_cut = int(train_ratio * 1000)
    val_cut = train_cut + int(val_ratio * 1000)
    if digest < train_cut:
        return "train"
    if digest < val_cut:
        return "val"
    return "test"


def _violations(assigned: list[tuple[str, list[str], str]], kind: str) -> int:
    """Count keys of one kind that appear in more than one split."""
    seen: dict[str, set[str]] = {}
    for _, keys, split in assigned:
        for key in keys:
            if _kind_key(key) == kind:
                seen.setdefault(key, set()).add(split)
    return sum(1 for splits in seen.values() if len(splits) > 1)


def _count(assigned: list[tuple[str, list[str], str]], kind: str) -> int:
    return len({key for _, keys, _ in assigned for key in keys if _kind_key(key) == kind})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    links = _Links()
    rows: list[tuple[str, list[str], bool]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            keys = _group_keys(record, args.seed)
            for key in keys[1:]:
                links.union(keys[0], key)
            rows.append((line, keys, _is_real_home(record)))

    units: dict[str, str] = {}
    assigned: list[tuple[str, list[str], str]] = []
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    real_by_split: dict[str, int] = {split: 0 for split in buckets}
    for line, keys, real in rows:
        root = links.find(keys[0])
        if root not in units:
            units[root] = assign_split(root)
        split = units[root]
        assigned.append((line, keys, split))
        buckets[split].append(line)
        real_by_split[split] += real

    print(f"Groups: {len({keys[0] for _, keys, _ in assigned})}")
    print(f"Households: {_count(assigned, 'household')}")
    print("Real-home rows: " + ", ".join(f"{split}={real_by_split[split]}" for split in buckets))
    group_violations = _violations(assigned, "contrast") + _violations(assigned, "scenario")
    household_violations = _violations(assigned, "household")
    print(f"Group violations: {group_violations}")
    print(f"Household violations: {household_violations}")
    if group_violations or household_violations:
        print("error: a group or household spans more than one split", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, lines in buckets.items():
        out = args.out_dir / f"sayso_{split}.jsonl"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{split}: {len(lines)} examples -> {out}")

    print(f"Families: {len(units)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
