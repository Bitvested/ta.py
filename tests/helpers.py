"""Float tolerance helpers. Mirrors ta.js's test/helpers/close.js."""
import math
from typing import Any

ABS_TOL = 1e-9
REL_TOL = 1e-9


def close_scalar(a: float, b: float) -> bool:
    """Return True if a and b are close within combined abs+rel tolerance. NaN-aware."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
    if a == b:
        return True
    diff = abs(a - b)
    if diff <= ABS_TOL:
        return True
    return diff <= REL_TOL * max(abs(a), abs(b))


def close(a: Any, b: Any) -> bool:
    """Deep-compare two values (scalars, lists, tuples, dicts) within tolerance."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return close_scalar(float(a), float(b))
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(close(a[k], b[k]) for k in a)
    return a == b


def assert_close(actual: Any, expected: Any, name: str = "") -> None:
    """Pytest-friendly assertion that prints the offending values."""
    if not close(actual, expected):
        prefix = f"{name}: " if name else ""
        raise AssertionError(f"{prefix}values not close.\n  actual:   {actual!r}\n  expected: {expected!r}")
