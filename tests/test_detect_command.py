"""Unit tests for detect_command — slash command parser."""

import pytest

from app.commands.handler import detect_command


@pytest.mark.parametrize(
    "text, expected",
    [
        # known commands, no body
        ("/whoami", ("whoami", "")),
        ("/WHOAMI", ("whoami", "")),
        ("  /whoami  ", ("whoami", "")),
        ("/whoami.", ("whoami", "")),
        ("/whoami!", ("whoami", "")),
        ("/help", ("help", "")),
        ("/hire-mark", ("hire-mark", "")),
        ("/resume", ("resume", "")),
        # /jdfit without body
        ("/jdfit", ("jdfit", "")),
        ("/jdfit   ", ("jdfit", "")),
        # /jdfit with body
        ("/jdfit some job description here", ("jdfit", "some job description here")),
        ("/jdfit  leading spaces in body", ("jdfit", "leading spaces in body")),
        # unknown slash command
        ("/unknown", ("unknown", "")),
        ("/foo-bar", ("foo-bar", "")),
        # not a command — returns None
        ("plain text", None),
        ("whoami", None),
        ("sudo hire-mark", None),
        ("cat resume.pdf", None),
        ("", None),
        ("  ", None),
        ("tell me about mark", None),
    ],
)
def test_detect_command(text: str, expected: tuple | None) -> None:
    assert detect_command(text) == expected
