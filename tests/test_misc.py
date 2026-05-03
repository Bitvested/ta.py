"""Tests for miscellaneous functions."""
import pytest
import ta_py
from tests import parameters as p
from tests.helpers import assert_close

CASES = ["times_up", "times_down"]


@pytest.mark.parametrize("name", CASES)
def test_misc(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)


def test_fibnumbers() -> None:
    assert ta_py.fibnumbers == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
