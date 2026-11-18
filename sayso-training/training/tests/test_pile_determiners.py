from __future__ import annotations

from generators.utterances import _drop_leading_determiner


def _render(template: str, slot: str, value: str) -> str:
    return template.replace("{" + slot + "}", _drop_leading_determiner(template, slot, value)).strip()


def test_drop_doubled_the_before_name_slot():
    assert _render("make the {name} spin", "name", "the Hallway fan") == "make the Hallway fan spin"


def test_drop_possessive_before_name_slot():
    assert _render("turn on my {name}", "name", "my Office Lamp") == "turn on my Office Lamp"


def test_keep_value_when_template_has_no_determiner_before_slot():
    assert _render("turn on {name}", "name", "the Office Lamp") == "turn on the Office Lamp"


def test_keep_value_that_does_not_start_with_a_determiner():
    assert _render("pause the {name}", "name", "Theater TV") == "pause the Theater TV"
