"""A new turn must silence the previous reply.

Streamlit keeps already-rendered elements until they are replaced, so the
previous answer's <audio autoplay> kept playing over the next question while
it was still being processed. `process()` clears the response placeholder and
drops the pending audio before any work starts.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"
SOURCE = APP.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in app.py")


def test_process_takes_the_response_placeholder():
    args = [a.arg for a in _function("process").args.args]
    assert "response_ph" in args


def test_previous_answer_is_cleared_before_any_processing():
    """Order matters: clearing after the LLM call would leave audio playing."""
    body = _function("process").body
    statements = [ast.dump(node) for node in body]

    cleared_at = next(
        i for i, dumped in enumerate(statements) if "response_ph" in dumped and "empty" in dumped
    )
    worked_at = next(
        i for i, dumped in enumerate(statements)
        if "handle" in dumped or "'thinking'" in dumped
    )
    assert cleared_at < worked_at, "the old answer is cleared too late to stop its audio"


def test_pending_audio_is_dropped_on_a_new_turn():
    body = ast.dump(ast.Module(body=_function("process").body, type_ignores=[]))
    assert "_pending_audio" in body


def test_response_is_rendered_inside_the_placeholder():
    """Otherwise clearing the placeholder would not remove the audio element."""
    assert "response_ph.container()" in SOURCE


def test_every_process_call_passes_the_placeholder():
    calls = [
        node for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "process"
    ]
    assert calls, "no process() calls found"
    for call in calls:
        assert len(call.args) == 3, f"process() called with {len(call.args)} args, expected 3"
