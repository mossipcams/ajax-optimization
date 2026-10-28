
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_split_assigns_whole_families(tmp_path: Path) -> None:
    records = []
    for template in ("turn_on", "turn_off"):
        for seed in (1, 2):
            records.append(
                {
                    "messages": [{"role": "user", "content": f"{template}-{seed}"}],
                    "metadata": {
                        "template_family": template,
                        "phrasing_family": "direct",
                        "seed": seed,
                    },
                }
            )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "split"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_dataset.py"),
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
    )
    train = (out_dir / "sayso_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val = (out_dir / "sayso_val.jsonl").read_text(encoding="utf-8").strip().splitlines()
    test = (out_dir / "sayso_test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train) + len(val) + len(test) == len(records)
    def families(lines: list[str]) -> set[str]:
        result = set()
        for line in lines:
            meta = json.loads(line).get("metadata", {})
            result.add(
                f"{meta.get('template_family')}|{meta.get('phrasing_family')}|{meta.get('seed')}"
            )
        return result

    assert not (families(train) & families(val))
    assert not (families(train) & families(test))
    assert not (families(val) & families(test))


def _system(entities: list[tuple[str, str, str]]) -> str:
    lines = [
        "You are SaySo, a local Home Assistant voice agent.",
        "Static Context: An overview of the areas and the devices in this smart home:",
    ]
    for names, domain, area in entities:
        lines.append(f"- names: {names}")
        lines.append(f"  domain: {domain}")
        if area:
            lines.append(f"  areas: {area}")
    lines += [
        "",
        "When controlling Home Assistant always call the intent tools.",
        "Area context:",
        "satellite_area: Living Room",
    ]
    return "\n".join(lines)


def _row(label, utterance, entities, *, metadata=None):
    return {
        "_label": label,
        "messages": [
            {"role": "system", "content": _system(entities)},
            {"role": "user", "content": utterance},
            {"role": "assistant", "content": "Done."},
        ],
        "metadata": metadata or {},
    }


def _split(tmp_path: Path, records: list[dict], *, name: str = "input.jsonl", seed: int = 42):
    input_path = tmp_path / name
    input_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out_dir = tmp_path / f"out-{name}-{seed}"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_dataset.py"),
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    buckets = {
        split: [
            json.loads(line)
            for line in (out_dir / f"sayso_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for split in ("train", "val", "test")
    }
    return buckets, result.stdout


def _splits_by(buckets: dict[str, list[dict]], key) -> dict:
    where: dict = {}
    for split, rows in buckets.items():
        for row in rows:
            where.setdefault(key(row), set()).add(split)
    return where


def _household_entities(index: int) -> list[tuple[str, str, str]]:
    return [
        (f"Lamp {index}, Lamp {index} light", "light", f"Room {index}"),
        (f"Speaker {index}", "media_player", f"Room {index}"),
        (f"Front Door {index}", "lock", None),
    ]


def test_contrast_variants_land_in_one_split(tmp_path: Path) -> None:
    records = []
    for group in range(30):
        entities = _household_entities(group)
        for variant in range(4):
            records.append(
                _row(
                    f"g{group}-v{variant}",
                    f"turn on lamp {group} please, take {variant}",
                    entities[: 1 + variant % 3] + [(f"Extra {group}-{variant}", "switch", None)],
                    metadata={
                        "family": "contrast",
                        "contrast_group": f"contrast_{group:06d}_lamp",
                        "contrast_axis": ("base", "presence", "count", "area")[variant],
                        "contrast_variant": variant,
                        "home_id": f"contrast_{group:06d}_lamp_{variant}",
                        "semantic_id": f"contrast_{group:06d}_lamp",
                    },
                )
            )
    buckets, stdout = _split(tmp_path, records)
    assert sum(len(rows) for rows in buckets.values()) == len(records)
    where = _splits_by(buckets, lambda row: row["metadata"]["contrast_group"])
    assert all(len(splits) == 1 for splits in where.values()), where
    assert "Group violations: 0" in stdout


def test_shared_household_lands_in_one_split(tmp_path: Path) -> None:
    records = []
    for household in range(40):
        entities = _household_entities(household)
        for copy in range(2):
            records.append(
                _row(
                    f"h{household}-{copy}",
                    f"switch the lamp {household} {copy}",
                    entities,
                    metadata={
                        "family": "direct",
                        "home_id": f"home_{household:06d}_{copy}",
                        "semantic_id": f"sem_{household}_{copy}",
                    },
                )
            )
    buckets, stdout = _split(tmp_path, records)
    where = _splits_by(buckets, lambda row: row["_label"].split("-")[0])
    assert all(len(splits) == 1 for splits in where.values()), where
    assert "Household violations: 0" in stdout
    assert "Households: 40" in stdout


def test_entity_order_does_not_change_household(tmp_path: Path) -> None:
    records = []
    for household in range(40):
        entities = _household_entities(household)
        for copy, order in enumerate((entities, list(reversed(entities)))):
            records.append(_row(f"h{household}", f"lamp {household} {copy}", order))
    buckets, stdout = _split(tmp_path, records)
    where = _splits_by(buckets, lambda row: row["_label"])
    assert all(len(splits) == 1 for splits in where.values()), where
    assert "Households: 40" in stdout


def test_scenario_rows_ignore_paraphrase_text(tmp_path: Path) -> None:
    records = []
    for scenario in range(30):
        for phrasing in ("please dim the lamp", "lamp to dim, thanks", "dim it"):
            records.append(
                _row(
                    f"s{scenario}",
                    f"{phrasing} {scenario}",
                    [(f"Scenario Lamp {scenario}", "light", f"Room {scenario}")],
                    metadata={
                        "family": "direct",
                        "home_id": f"home_{scenario:06d}_32",
                        "semantic_id": f"sem_{scenario:06d}",
                    },
                )
            )
    buckets, _ = _split(tmp_path, records)
    where = _splits_by(buckets, lambda row: row["_label"])
    assert all(len(splits) == 1 for splits in where.values()), where


def test_split_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    records = [
        _row(
            f"d{index}",
            f"lamp {index}",
            _household_entities(index % 25),
            metadata={"semantic_id": f"sem_{index}", "home_id": f"home_{index}"},
        )
        for index in range(120)
    ]
    first, _ = _split(tmp_path, records, name="first.jsonl")
    second, _ = _split(tmp_path, records, name="second.jsonl")
    assert first == second

    shuffled = list(records)
    random.Random(1).shuffle(shuffled)
    reordered, _ = _split(tmp_path, shuffled, name="shuffled.jsonl")
    assert {split: {row["_label"] for row in rows} for split, rows in reordered.items()} == {
        split: {row["_label"] for row in rows} for split, rows in first.items()
    }


def test_legacy_rows_without_new_metadata_still_split(tmp_path: Path) -> None:
    records = [
        {
            "messages": [{"role": "user", "content": f"legacy utterance {index}"}],
            "metadata": {
                "template_family": f"template_{index % 7}",
                "phrasing_family": "direct",
                "seed": index,
            },
        }
        for index in range(60)
    ]
    records += [
        {"messages": [{"role": "user", "content": f"bare utterance {index}"}]} for index in range(40)
    ]
    buckets, stdout = _split(tmp_path, records)
    assert sum(len(rows) for rows in buckets.values()) == len(records)
    assert buckets["train"]
    assert "Group violations: 0" in stdout
    assert "Household violations: 0" in stdout


def test_household_signature_reads_wrapped_entities() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("split_dataset", ROOT / "scripts" / "split_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wrapped = [
        "- names: Living room table lamp, Living Room light, Living room table light, side",
        "    lamp",
        "  domain: light",
        "  areas: Living Room",
        "- names: Hall Plug",
        "  domain: switch",
    ]
    text = "Static Context: An overview of the areas and the devices:\n" + "\n".join(wrapped)
    text += "\n\nWhen controlling Home Assistant always call the intent tools.\n"

    def signature(system: str) -> str | None:
        return module._household_signature({"messages": [{"role": "system", "content": system}]})

    full = signature(text)
    assert full is not None
    hall = "- names: Hall Plug\n  domain: switch"
    assert signature(text.replace("\n" + hall, "")) != full
    reordered = text.replace("\n" + hall, "").replace(wrapped[0], hall + "\n" + wrapped[0])
    assert signature(reordered) == full


def _real_home_row(index: int) -> dict:
    return _row(
        f"real-{index}",
        f"real home utterance {index}",
        _household_entities(0),
        metadata={
            "real_home": True,
            "home_id": "real_home_train",
            "semantic_id": f"real_sem_{index:04d}",
        },
    )


def test_real_home_rows_spread_across_splits(tmp_path: Path) -> None:
    records = [_real_home_row(index) for index in range(200)]
    buckets, stdout = _split(tmp_path, records)
    assert sum(len(rows) for rows in buckets.values()) == len(records)
    counts = {split: len(rows) for split, rows in buckets.items()}
    assert counts["train"] >= 120, counts
    assert 5 <= counts["val"] <= 40, counts
    assert 5 <= counts["test"] <= 40, counts
    assert f"Real-home rows: train={counts['train']}, val={counts['val']}, test={counts['test']}" in stdout
    where = _splits_by(buckets, lambda row: row["_label"])
    assert all(len(splits) == 1 for splits in where.values()), where


def test_real_home_rows_do_not_count_as_households(tmp_path: Path) -> None:
    records = [_real_home_row(index) for index in range(120)]
    for household in range(40):
        entities = _household_entities(household + 1)
        for copy in range(2):
            records.append(
                _row(
                    f"h{household}-{copy}",
                    f"switch the lamp {household} {copy}",
                    entities,
                    metadata={
                        "family": "direct",
                        "home_id": f"home_{household:06d}_{copy}",
                        "semantic_id": f"sem_{household}_{copy}",
                    },
                )
            )
    buckets, stdout = _split(tmp_path, records)
    where = _splits_by(buckets, lambda row: row["_label"].split("-")[0])
    household_where = {
        label: splits for label, splits in where.items() if not label.startswith("real")
    }
    assert all(len(splits) == 1 for splits in household_where.values()), household_where
    assert "Households: 40" in stdout
    assert "Household violations: 0" in stdout
    assert "Real-home rows:" in stdout
