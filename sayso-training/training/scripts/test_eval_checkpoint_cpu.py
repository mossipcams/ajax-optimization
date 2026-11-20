from __future__ import annotations

import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from evals.cases import select_cases, load_all_cases
from scripts.eval_checkpoint_cpu import suite_cases


def test_regression_suite_uses_regression_tagged_cases():
    cases = suite_cases("regression")
    assert [case.id for case in cases] == [case.id for case in select_cases(tag="regression")]
    assert len(cases) > 0
    assert all("regression" in case.tags for case in cases)


def test_regression_suite_contains_all_regression_tagged_cases():
    tagged = {case.id for case in load_all_cases() if "regression" in case.tags}
    assert tagged <= {case.id for case in suite_cases("regression")}


def test_smoke_suite_unchanged():
    cases = suite_cases("smoke")
    assert [case.id for case in cases] == [case.id for case in select_cases(suite="smoke")]
