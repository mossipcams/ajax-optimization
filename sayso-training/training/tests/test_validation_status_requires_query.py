import random

from generators.homes import generate_home
from generators.tools import build_query
from generators.validation import validate_spec


def _spec(home, expected):
    return {"home": home, "expected": expected}


def test_status_row_without_get_live_context_is_rejected():
    home = generate_home(3, 16, random.Random(3))
    spec = _spec(home, {"kind": "status", "calls": []})
    assert validate_spec(spec) == "status_requires_state_query"


def test_status_row_with_get_live_context_is_accepted():
    home = generate_home(4, 16, random.Random(4))
    entity = home["entities"][0]
    spec = _spec(home, {"kind": "status", "calls": [build_query(entity)]})
    assert validate_spec(spec) is None
