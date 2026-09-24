"""Turn findings into terminal output or JSON."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

from .checks import FAIL, PASS, WARN, Finding

_SYMBOLS = {PASS: ("✓", "32"), WARN: ("⚠", "33"), FAIL: ("✗", "31")}


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def to_text(findings: list[Finding], train_shape: tuple[int, int], test_shape: tuple[int, int]) -> str:
    lines = [f"leakcheck  train: {train_shape[0]:,} rows × {train_shape[1]} cols   "
             f"test: {test_shape[0]:,} rows × {test_shape[1]} cols", ""]
    order = {FAIL: 0, WARN: 1, PASS: 2}
    for f in sorted(findings, key=lambda f: order[f.status]):
        symbol, code = _SYMBOLS[f.status]
        lines.append(f"  {_color(symbol, code)} {f.message}")

    fails = sum(f.status == FAIL for f in findings)
    warns = sum(f.status == WARN for f in findings)
    lines.append("")
    if fails:
        lines.append(_color(f"{fails} problem(s), {warns} warning(s) — likely leakage", "31;1"))
    elif warns:
        lines.append(_color(f"No problems, {warns} warning(s) worth a look", "33;1"))
    else:
        lines.append(_color("All clear — no leakage detected", "32;1"))
    return "\n".join(lines)


def to_json(findings: list[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], indent=2, default=str)
