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


def test_random_namespace_exposes_prng() -> None:
    assert ta_py.random.prng is ta_py.prng


def test_random_pick_matches_tajs() -> None:
    r = ta_py.random.prng("test-seed")
    assert ta_py.random.pick([1, 2, 3, 4, 5], r) == 2
    assert ta_py.random.pick([1, 2, 3, 4, 5], r) == 3
    assert ta_py.random.pick([1, 2, 3, 4, 5], r) == 2


def test_random_range_matches_tajs() -> None:
    r = ta_py.random.prng("test-seed")
    assert ta_py.random.range(1, 10, r) == 4
    assert ta_py.random.range(1, 10, r) == 4
    assert ta_py.random.range(1, 10, r) == 3


def test_random_float_matches_tajs() -> None:
    r = ta_py.random.prng("test-seed")
    assert ta_py.random.float(1.5, 2.5, r) == 1.8591179228387773
    assert ta_py.random.float(1.5, 2.5, r) == 1.9248472950421274


def test_random_order_matches_tajs() -> None:
    r = ta_py.random.prng("test-seed")
    assert ta_py.random.order([1, 2, 3, 4, 5], r) == [2, 3, 1, 4, 5]
