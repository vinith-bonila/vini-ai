"""
Calculator and unit conversion.

Arithmetic is evaluated by walking a restricted AST - never `eval()` - so
arbitrary code cannot run. Natural-language operators ("times", "percent of")
are normalised to symbols first.
"""
from __future__ import annotations

import ast
import operator
import re

from assistant.schemas import SkillResponse
from utils.logger import get_logger

logger = get_logger(__name__)

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}

_WORDS = [
    (r"\bplus\b|\band\b", "+"), (r"\bminus\b", "-"), (r"\btimes\b|\bmultiplied by\b|\bx\b", "*"),
    (r"\bdivided by\b|\bover\b", "/"), (r"\bto the power of\b|\bpower\b|\braised to\b", "**"),
    (r"\bmod\b|\bmodulo\b", "%"),
]

_PERCENT_OF = re.compile(r"(?P<a>[\d.]+)\s*percent\s+of\s+(?P<b>[\d.]+)", re.I)


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _normalise(text: str) -> str:
    expr = text.lower()
    m = _PERCENT_OF.search(expr)
    if m:
        expr = _PERCENT_OF.sub(f"({m.group('a')}/100*{m.group('b')})", expr)
    expr = re.sub(r"\bpercent\b", "/100", expr)
    for pattern, sym in _WORDS:
        expr = re.sub(pattern, sym, expr)
    for stop in ("what is", "calculate", "compute", "how much is", "whats", "="):
        expr = expr.replace(stop, "")
    expr = re.sub(r"[^0-9+\-*/%.()\s]", "", expr)
    return expr.strip()


def calculate(text: str, entities=None, context=None) -> SkillResponse:
    expr = _normalise(text)
    if not expr or not any(c.isdigit() for c in expr):
        return SkillResponse.error("Give me an arithmetic expression to calculate.")
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
        result = round(result, 6)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return SkillResponse(speech=f"The answer is {result}.", data={"expression": expr, "result": result})
    except Exception as exc:  # noqa: BLE001
        logger.debug("Calc error for '%s': %s", expr, exc)
        return SkillResponse.error("I couldn't work that out. Try phrasing it as numbers and operators.")


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #
_LINEAR = {  # to a common base unit
    ("km", "mi"): lambda v: v * 0.621371, ("mi", "km"): lambda v: v / 0.621371,
    ("kg", "lb"): lambda v: v * 2.20462, ("lb", "kg"): lambda v: v / 2.20462,
    ("g", "oz"): lambda v: v * 0.035274, ("oz", "g"): lambda v: v / 0.035274,
    ("m", "ft"): lambda v: v * 3.28084, ("ft", "m"): lambda v: v / 3.28084,
    ("hours", "minutes"): lambda v: v * 60, ("minutes", "hours"): lambda v: v / 60,
    ("c", "f"): lambda v: v * 9 / 5 + 32, ("f", "c"): lambda v: (v - 32) * 5 / 9,
}
_ALIAS = {
    "kilometers": "km", "kilometres": "km", "km": "km", "miles": "mi", "mile": "mi",
    "kilograms": "kg", "kg": "kg", "pounds": "lb", "lb": "lb", "grams": "g", "g": "g",
    "ounces": "oz", "oz": "oz", "meters": "m", "metres": "m", "m": "m", "feet": "ft", "ft": "ft",
    "hours": "hours", "hour": "hours", "minutes": "minutes", "minute": "minutes",
    "celsius": "c", "fahrenheit": "f",
}
_CONV = re.compile(r"(?P<val>[\d.]+)\s*(?P<from>[a-zA-Z]+)\s+(?:to|in|into)\s+(?P<to>[a-zA-Z]+)", re.I)


def convert_units(text: str, entities=None, context=None) -> SkillResponse:
    m = _CONV.search(text)
    if not m:
        return SkillResponse.error("Try something like 'convert 10 km to miles'.")
    val = float(m.group("val"))
    u_from = _ALIAS.get(m.group("from").lower())
    u_to = _ALIAS.get(m.group("to").lower())
    fn = _LINEAR.get((u_from, u_to))
    if not fn:
        return SkillResponse.error(f"I can't convert {m.group('from')} to {m.group('to')} yet.")
    out = round(fn(val), 4)
    return SkillResponse(speech=f"{val} {m.group('from')} is {out} {m.group('to')}.",
                         data={"value": val, "from": u_from, "to": u_to, "result": out})
