import json
from pathlib import Path

from generators.capability_registry import CAPABILITIES
from generators.config import GeneratorConfig, default_allocations
from generators.context import serialize_context
from generators.pipeline import run_generation
from generators.planning import REAL_HOME_REFUSAL_SHARE_CAP

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_reference_home.json"


def test_real_home_absence_and_junk_keep_the_production_context():
    result = run_generation(GeneratorConfig(
        count=1500, seed=515, real_home_path=FIXTURE, real_home_rate=0.25,
        paraphrase_enabled=False, allocations=default_allocations(),
    ))
    real = result["stats"]["real_home"]
    rows = [row for row in result["rows"] if row["metadata"]["real_home"]]
    refusals = [row for row in rows if row["metadata"].get("family") in {"absence", "junk"}]
    assert {row["metadata"]["family"] for row in refusals} == {"absence", "junk"}
    assert real["refusal_rows"] == len(refusals)
    assert len(refusals) <= REAL_HOME_REFUSAL_SHARE_CAP * real["rows"]

    home = json.loads(FIXTURE.read_text())
    for row in refusals:
        assert not any(message.get("tool_calls") for message in row["messages"])
        # The full real home, nothing removed: byte-identical to production.
        assert row["messages"][0]["content"] == serialize_context(home, row["messages"][1]["content"])
        if row["metadata"]["family"] == "absence":
            domain = CAPABILITIES[row["metadata"]["capability"]].domain
            area = next(
                area for area in home["areas"]
                if f"The {area.lower()} has no" in row["messages"][-1]["content"]
            )
            assert not any(e["domain"] == domain and e["area"] == area for e in home["entities"])
