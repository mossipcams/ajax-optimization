from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from generators.config import GeneratorConfig
from generators.pipeline import run_generation

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SMOKE = ROOT / "configs" / "generation" / "smoke.yaml"
GOLDEN_SEED = 20260917


def _sorted(counts: dict) -> dict:
    return dict(sorted(counts.items()))


@pytest.mark.slow
def test_smoke_generation_golden_summary(snapshot) -> None:
    config = GeneratorConfig.from_yaml(SMOKE, repo_root=REPO)
    config.seed = GOLDEN_SEED
    result = run_generation(config)
    rows = result["rows"]
    manifest = result["manifest"]
    families = Counter(str(row["metadata"].get("family") or "none") for row in rows)
    summary = {
        "row_count": len(rows),
        "by_family": _sorted(families),
        "by_tool": manifest["positive_tool_coverage"],
        "by_outcome": manifest["outcome_coverage"],
        "rejection_reasons": manifest["rejections"],
        "dataset_hash": manifest["dataset_hash"],
    }
    assert summary == snapshot
