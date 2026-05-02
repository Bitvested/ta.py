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
