import pytest

from generators.validation import validate_utterance

HALF_CASES = [
    ("set a timer for an hour and a half", {"hours": 1, "minutes": 30}),
    ("set a timer for one and a half hours", {"hours": 1, "minutes": 30}),
    ("set a timer for half an hour", {"minutes": 30}),
    ("set a timer for a minute and a half", {"minutes": 1, "seconds": 30}),
]


def _spec(utterance, arguments):
    return {
        "utterance": utterance,
        "expected": {"kind": "action", "calls": [{"name": "HassStartTimer", "arguments": arguments}]},
    }


@pytest.mark.parametrize(("utterance", "arguments"), HALF_CASES)
def test_half_phrase_validates_for_its_own_shape(utterance, arguments):
    assert validate_utterance(_spec(utterance, arguments)) is None


def test_half_an_hour_does_not_satisfy_hours_and_minutes_shape():
    assert validate_utterance(_spec("set a timer for half an hour", {"hours": 1, "minutes": 30})) == "missing_expected_value"


def test_an_hour_alone_does_not_satisfy_hours_and_minutes_shape():
    assert validate_utterance(_spec("set a timer for an hour", {"hours": 1, "minutes": 30})) == "missing_expected_value"


def test_half_phrase_does_not_satisfy_other_numeric_keys():
    assert validate_utterance(_spec("set a timer for an hour and a half", {"hours": 2, "minutes": 30})) == "missing_expected_value"


def test_digit_forms_still_validate():
    assert validate_utterance(_spec("set a timer for 1 hour and 30 minutes", {"hours": 1, "minutes": 30})) is None
    assert validate_utterance(_spec("set a timer for 30 minutes", {"minutes": 30})) is None
    assert validate_utterance(_spec("set a timer for 1 minute and 30 seconds", {"minutes": 1, "seconds": 30})) is None
