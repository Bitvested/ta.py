"""Tests for bands functions."""
import pytest
import ta_py
from tests import parameters as p
from tests.helpers import assert_close

CASES = ["bands", "bandwidth", "keltner", "don", "envelope", "fibbands"]


@pytest.mark.parametrize("name", CASES)
def test_band(name: str) -> None:
    fixture = getattr(p, name)
    fn = getattr(ta_py, name)
    actual = fn(*fixture["in"])
    assert_close(actual, fixture["out"], name=name)
