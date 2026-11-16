import re

from generators.ohf import render_call

HALF_SHAPES = (
    ({"hours": 1, "minutes": 30}, ("an hour and a half", "one and a half hours")),
    ({"minutes": 30}, ("half an hour",)),
    ({"minutes": 1, "seconds": 30}, ("a minute and a half",)),
)

OTHER_DURATIONS = (
    {"minutes": 45},
    {"hours": 2, "minutes": 30},
    {"hours": 1, "minutes": 45},
    {"minutes": 1, "seconds": 45},
    {"seconds": 30},
    {"hours": 1},
)

HALF_PATTERN = re.compile(r"\bhalf\b|\band a half\b")


def _texts(arguments, seeds=range(600)):
    call = {"name": "HassStartTimer", "arguments": arguments}
    return [text for text in (render_call(call, "", seed) for seed in seeds) if text is not None]


def test_half_forms_render_for_each_half_shape():
    for arguments, phrases in HALF_SHAPES:
        texts = _texts(arguments)
        for phrase in phrases:
            assert any(phrase in text for text in texts), (phrase, texts[:20])


def test_half_forms_follow_create_a_timer_for():
    for arguments, phrases in HALF_SHAPES:
        texts = _texts(arguments)
        assert any(
            re.search(r"create a timer for " + re.escape(phrase), text)
            for phrase in phrases
            for text in texts
        ), (arguments, texts[:20])


def test_half_forms_never_render_for_other_durations():
    for arguments in OTHER_DURATIONS:
        for text in _texts(arguments):
            assert not HALF_PATTERN.search(text), (arguments, text)


def test_half_forms_do_not_cross_between_shapes():
    for arguments, phrases in HALF_SHAPES:
        own = set(phrases)
        others = {phrase for _, other in HALF_SHAPES if other != phrases for phrase in other}
        for text in _texts(arguments):
            assert not any(phrase in text for phrase in others), (arguments, text)
            if HALF_PATTERN.search(text):
                assert any(phrase in text for phrase in own), (arguments, text)


def test_half_utterances_keep_no_digits_for_the_duration():
    for arguments, _ in HALF_SHAPES:
        for text in _texts(arguments):
            if HALF_PATTERN.search(text):
                assert not re.search(r"\d", text), text
