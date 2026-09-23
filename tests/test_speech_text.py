"""Markdown must never be spoken aloud.

LLM replies come back as Markdown, so text-to-speech was reading the syntax:
'#' as "hash", '**' as "asterisk asterisk", and a table as a run of pipes.
"""
from __future__ import annotations

import pytest

from utils.helpers import speech_text


@pytest.mark.parametrize(
    "markdown, spoken",
    [
        ("# Heading", "Heading"),
        ("### Deep heading", "Deep heading"),
        ("**bold**", "bold"),
        ("*italic*", "italic"),
        ("***both***", "both"),
        ("__strong__", "strong"),
        ("`inline code`", "inline code"),
        ("> quoted line", "quoted line"),
        (">> double quoted", "double quoted"),
        ("- bullet", "bullet"),
        ("* star bullet", "star bullet"),
        ("1. first", "first"),
        ("2) second", "second"),
    ],
)
def test_syntax_is_removed_but_words_survive(markdown, spoken):
    assert speech_text(markdown) == spoken


def test_links_keep_their_label_and_drop_the_url():
    result = speech_text("See [the official site](https://iipe.ac.in/) now")
    assert "the official site" in result
    assert "iipe.ac.in" not in result
    assert "](" not in result


def test_images_are_dropped_entirely():
    assert "logo.png" not in speech_text("![a logo](logo.png) hello")


def test_code_blocks_are_not_read_aloud():
    result = speech_text('Before\n```python\nprint("secret")\n```\nAfter')
    assert "secret" not in result
    assert "Before" in result and "After" in result


def test_tables_lose_their_pipes():
    result = speech_text("| Item | Detail |\n|------|--------|\n| Type | Private |")
    assert "|" not in result
    assert "-----" not in result
    assert "Type" in result and "Private" in result


def test_horizontal_rules_are_dropped():
    assert "---" not in speech_text("one\n\n---\n\ntwo")


def test_a_full_reply_contains_no_markdown_characters():
    reply = (
        "# IIPE - Overview\n\n> **IIPE** is an *institute* in **Visakhapatnam**.\n\n"
        "| Item | Detail |\n|---|---|\n| Founded | 2016 |\n\n"
        "- Point one\n- Point two\n\n---\n\nSee [the site](https://iipe.ac.in/).\n"
    )
    spoken = speech_text(reply)
    for junk in ("#", "*", "|", "`", "---", "](", ">"):
        assert junk not in spoken, f"{junk!r} would be read aloud"
    assert "Visakhapatnam" in spoken


def test_plain_text_is_left_alone():
    plain = "IIPE is located in Visakhapatnam, Andhra Pradesh."
    assert speech_text(plain) == plain


@pytest.mark.parametrize("empty", ["", "   ", "\n\n", None])
def test_empty_input_is_safe(empty):
    assert speech_text(empty) == ""


def test_asterisk_in_prose_is_not_treated_as_emphasis():
    assert "2 * 3" in speech_text("The product 2 * 3 equals 6.")
