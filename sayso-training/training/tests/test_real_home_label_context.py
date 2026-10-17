import json
from pathlib import Path

import pytest

from generators.config import GeneratorConfig, default_allocations
from generators.context import serialize_context
from generators.pipeline import run_generation
from generators.real_home import holdout_entity_names

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_home.json"


def _normalized_home():
    home = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for entity in [*home["entities"], *home.get("context_entities", [])]:
        entity["name"] = " ".join(entity["name"].split())
        if entity.get("aliases"):
            entity["aliases"] = [" ".join(alias.split()) for alias in entity["aliases"]]
    return home


def _names(message):
    for call in message.get("tool_calls") or []:
        name = json.loads(call["function"]["arguments"]).get("name")
        if name:
            yield call["function"]["name"].split("__")[-1], name


@pytest.fixture(scope="module")
def real_rows():
    result = run_generation(GeneratorConfig(
        count=900, seed=29, real_home_path=FIXTURE, real_home_rate=0.3,
        paraphrase_enabled=False, allocations=default_allocations(),
    ))
    rows = [row for row in result["rows"] if row["metadata"]["real_home"]]
    assert len(rows) > 100
    return rows


def test_real_home_rows_render_the_production_home_unchanged(real_rows):
    home = _normalized_home()
    for row in real_rows:
        # Byte-identical context means no synthetic entity was added and no
        # holdout was hidden when the label was decided.
        assert row["messages"][0]["content"] == serialize_context(home, row["messages"][1]["content"])


def test_holdouts_stay_visible_but_are_never_named_targets(real_rows):
    holdouts = holdout_entity_names(FIXTURE)
    assert all(name in real_rows[0]["messages"][0]["content"] for name in holdouts)
    for row in real_rows:
        for message in row["messages"]:
            assert not {name for _, name in _names(message)} & holdouts


def test_final_calls_name_only_entities_in_the_real_home(real_rows):
    home = _normalized_home()
    known = {entity["name"] for entity in home["entities"]}
    known |= {alias for entity in home["entities"] for alias in entity.get("aliases") or []}
    for row in real_rows:
        final = [m for m in row["messages"] if m.get("tool_calls")][-1:]
        for message in final:
            for tool, name in _names(message):
                if tool.startswith("Hass") and "Timer" not in tool and "ListAdd" not in tool:
                    assert name in known, (name, row["messages"][1]["content"])
