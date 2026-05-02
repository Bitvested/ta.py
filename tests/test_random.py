"""Tests for random/PRNG functions."""
import ta_py


def test_prng_returns_callable() -> None:
    r = ta_py.prng("abc")
    assert callable(r)


def test_prng_values_in_range() -> None:
    r = ta_py.prng("abc")
    v1, v2 = r(), r()
    assert 0 <= v1 < 1
    assert 0 <= v2 < 1


def test_prng_same_seed_same_sequence() -> None:
    r1 = ta_py.prng("abc")
    r2 = ta_py.prng("abc")
    assert r1() == r2()
    assert r1() == r2()


def test_prng_different_seeds_diverge() -> None:
    assert ta_py.prng("xyz")() != ta_py.prng("abc")()
