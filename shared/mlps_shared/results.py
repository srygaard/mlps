"""Standardised result printing for mlps benchmarks."""
from __future__ import annotations

from collections.abc import Iterable


def section(title: str) -> None:
    """Print a section header: === title ==="""
    print(f"=== {title} ===")


def kvrows(items: Iterable[tuple[str, str]], indent: int = 2) -> None:
    """Print aligned  key : value  rows."""
    pairs = list(items)
    if not pairs:
        return
    pad = " " * indent
    w = max(len(k) for k, _ in pairs)
    for k, v in pairs:
        print(f"{pad}{k:<{w}} : {v}")
