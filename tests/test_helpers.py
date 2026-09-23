"""Helper behaviour, chiefly that error text can never carry a credential.

`failure_reason` output is both shown in the UI and written to the turn log,
so a leaked key would be persisted to disk.
"""
from __future__ import annotations

import pytest

from utils.helpers import failure_reason, truncate


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefgh12345678",
        "gsk-TESTKEY1234567890abcdefXYZ",
        "sk-proj-AbCdEf123456789_xyz",
        "xai-0123456789abcdefghij",
    ],
)
def test_keys_are_redacted(secret):
    reason = failure_reason(RuntimeError(f"401 unauthorized: {secret} rejected"))
    assert secret not in reason
    assert "[redacted]" in reason


def test_diagnostic_detail_survives_redaction():
    reason = failure_reason(RuntimeError("Error code: 401 - invalid_api_key"))
    assert "401" in reason
    assert "RuntimeError" in reason


def test_multiline_errors_collapse_to_one_line():
    reason = failure_reason(ValueError("line one\nline two\n\tline three"))
    assert "\n" not in reason
    assert "line one line two line three" in reason


def test_reason_is_bounded():
    reason = failure_reason(RuntimeError("x" * 5000))
    assert len(reason) <= 141


def test_exception_with_no_message_still_names_the_type():
    assert failure_reason(TimeoutError()) == "TimeoutError"


def test_truncate_leaves_short_text_alone():
    assert truncate("hello", 80) == "hello"


def test_truncate_marks_elision():
    assert truncate("x" * 100, 20).endswith("…")
