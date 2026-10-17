from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "audit_v7_corpus.py"

_loader = importlib.machinery.SourceFileLoader("audit_v7_corpus", str(SCRIPT))
_spec = importlib.util.spec_from_loader("audit_v7_corpus", _loader)
assert _spec is not None
audit_v7_corpus = importlib.util.module_from_spec(_spec)
_loader.exec_module(audit_v7_corpus)


ENTITIES = [
    "Kitchen light",
    "Christmas tree",
    "Desk Echo Dot",
    "Rocky room cam",
    "Living room lights",
    "TV",
    "Bedroom fan",
    "Nightstand light",
    "Thermostat",
    "Porch light",
    "Washer dryer combo",
]

SYSTEM = "Available devices: Kitchen light, Christmas tree, Desk Echo Dot, Rocky room cam, Living room lights, TV, Bedroom fan, Nightstand light, Thermostat, Porch light, Washer dryer combo."


def make_row(user="turn it on", targets=(), meta=None):
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    for target in targets:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "light.turn_on", "arguments": json.dumps({"name": target})}}
                ],
            }
        )
        messages.append({"role": "tool", "content": "ok"})
    return {"messages": messages, "tools": [], "metadata": meta or {}}


def base_corpus():
    rows = [make_row(targets=(name,), meta={"real_home": True}) for name in ENTITIES]
    for i in range(300):
        group = f"cg-{i}"
        meta_a = {"pair_kind": "real_home_contrast", "contrastive_group": group}
        meta_b = dict(meta_a, contrastive_group=group)
        rows.append(make_row(user="which light?", targets=("Kitchen light",), meta=meta_a))
        rows.append(make_row(user="which one?", targets=("Bedroom fan",), meta=meta_b))
    for i in range(1000):
        rows.append(make_row(meta={"pair_kind": "withheld_tool"}))
        rows.append(make_row(meta={"pair_kind": "clarify"}))
    for i in range(500):
        rows.append(
            make_row(
                meta={
                    "excluded_names": ["Rocky room camera"],
                    "expected_target_names": ["Rocky room cam"],
                }
            )
        )
    return rows


def write_corpus(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    fixture = tmp_path / "real_home.json"
    fixture.write_text(
        json.dumps(
            {
                "entities": [{"name": name, "aliases": [name]} for name in ENTITIES],
                "context_entities": [{"name": "Mousetrap sensor"}],
            }
        ),
        encoding="utf-8",
    )
    return path, fixture


def run(tmp_path, rows):
    corpus, fixture = write_corpus(tmp_path, rows)
    return audit_v7_corpus.run_audit(corpus, fixture)


def test_all_gates_pass(tmp_path):
    report = run(tmp_path, base_corpus())
    assert report["passed"] is True
    for name, gate in report["gates"].items():
        assert gate["pass"] is True, f"gate {name} failed: {gate}"


def test_fabricated_targets_fails(tmp_path):
    rows = base_corpus()
    rows.append(make_row(targets=("Invented Device",)))
    report = run(tmp_path, rows)
    assert report["gates"]["fabricated_targets"] == {
        "value": 1,
        "threshold": 0,
        "pass": False,
    }
    assert report["passed"] is False


def test_context_only_actions_fails(tmp_path):
    rows = base_corpus()
    rows.append(make_row(targets=("Mousetrap sensor",), meta={"real_home": True}))
    report = run(tmp_path, rows)
    assert report["gates"]["context_only_actions"]["pass"] is False
    assert report["gates"]["context_only_actions"]["value"] == 1


def test_eval_overlap_fails(tmp_path):
    cases = importlib.import_module("evals.cases")

    utterance = sorted(cases.excluded_train_utterances())[0]
    rows = base_corpus()
    rows.append(make_row(user=utterance))
    report = run(tmp_path, rows)
    assert report["gates"]["eval_overlap"] == {
        "value": 1,
        "threshold": 0,
        "pass": False,
    }


def test_holdout_targets_fails(tmp_path):
    rows = [row for row in base_corpus() if not (row["metadata"].get("real_home") and "Kitchen light" in json.dumps(row["messages"]))]
    report = run(tmp_path, rows)
    gate = report["gates"]["holdout_targets"]
    assert gate["value"]["Kitchen light"] == 0
    assert gate["pass"] is False


def test_tv_vs_living_room_fails(tmp_path):
    rows = base_corpus()
    rows += [make_row(targets=("TV",), meta={"real_home": True}) for _ in range(3)]
    report = run(tmp_path, rows)
    gate = report["gates"]["tv_vs_living_room"]
    assert gate["value"] == {"TV": 4, "Living room lights": 1}
    assert gate["pass"] is False


def test_max_real_home_target_share_fails(tmp_path):
    rows = base_corpus()
    rows += [make_row(targets=("TV",), meta={"real_home": True}) for _ in range(20)]
    report = run(tmp_path, rows)
    gate = report["gates"]["max_real_home_target_share"]
    assert gate["value"] > 0.10
    assert gate["pass"] is False


def test_contrast_groups_below_threshold_fails(tmp_path):
    rows = [row for row in base_corpus() if row["metadata"].get("pair_kind") != "real_home_contrast"]
    report = run(tmp_path, rows)
    gate = report["gates"]["real_home_contrast_groups"]
    assert gate["value"]["groups"] == 0
    assert gate["pass"] is False


def test_contrast_group_system_mismatch_fails(tmp_path):
    rows = base_corpus()
    for row in rows:
        if row["metadata"].get("contrastive_group") == "cg-0":
            row["messages"][0] = {"role": "system", "content": SYSTEM + " changed"}
            break
    report = run(tmp_path, rows)
    gate = report["gates"]["real_home_contrast_groups"]
    assert gate["value"]["identical_systems"] is False
    assert gate["pass"] is False


def test_withheld_and_clarify_pairs_fails(tmp_path):
    rows = [row for row in base_corpus() if row["metadata"].get("pair_kind") not in ("withheld_tool", "clarify")]
    report = run(tmp_path, rows)
    assert report["gates"]["withheld_pairs"] == {"value": 0, "threshold": 1000, "pass": False}
    assert report["gates"]["clarify_pairs"] == {"value": 0, "threshold": 1000, "pass": False}


def test_nearest_name_exclusions_fails(tmp_path):
    rows = [row for row in base_corpus() if not (row["metadata"].get("excluded_names") and row["metadata"].get("expected_target_names"))]
    report = run(tmp_path, rows)
    assert report["gates"]["nearest_name_exclusions"] == {
        "value": 0,
        "threshold": 500,
        "pass": False,
    }


def test_nearest_name_ignores_stop_words(tmp_path):
    rows = base_corpus()
    for row in rows:
        meta = row["metadata"]
        if meta.get("excluded_names") == ["Rocky room camera"]:
            meta["excluded_names"] = ["The lamp"]
            break
    report = run(tmp_path, rows)
    assert report["gates"]["nearest_name_exclusions"]["value"] == 499


def test_normalize_collapses_whitespace(tmp_path):
    rows = base_corpus()
    for row in rows:
        if "Rocky room cam" in json.dumps(row["messages"]) and not row["metadata"].get("pair_kind"):
            for call in row["messages"][2]["tool_calls"]:
                call["function"]["arguments"] = json.dumps({"name": "  Rocky   room  cam "})
            break
    report = run(tmp_path, rows)
    assert report["gates"]["holdout_targets"]["value"]["Rocky room cam"] >= 1


def test_generator_config_loads_pairs(tmp_path):
    from generators.config import GeneratorConfig

    repo_root = HERE.parent.parent
    config = GeneratorConfig.from_yaml(
        repo_root / "training/configs/generation/full_sft_v7.yaml",
        repo_root=repo_root,
    )
    assert config.pair_rates == {
        "withheld_tool": 0.6,
        "clarify": 0.4,
        "real_home_contrast": 0.2,
    }
