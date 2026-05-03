"""Smoke tests for ta_py.multi.sim (multi-process Monte Carlo)."""
import ta_py


def test_multi_sim_returns_correct_shape() -> None:
    data = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0]
    sd = ta_py.multi.sim(data, l=5, sims=16)
    assert len(sd) == 16
    assert all(len(path) == len(data) + 5 for path in sd)
    # Original data is preserved as the prefix of each path
    for path in sd:
        assert path[: len(data)] == data


def test_multi_sim_with_percentile_returns_single_path() -> None:
    data = [100.0, 102.0, 101.0, 103.0, 105.0]
    final = ta_py.multi.sim(data, l=5, sims=16, perc=0.5)
    assert len(final) == len(data) + 5
    assert final[: len(data)] == data


def test_multi_sim_falls_back_for_small_sims() -> None:
    """sims < 8 short-circuits to the sequential implementation; output stays valid."""
    data = [100.0, 102.0, 101.0, 103.0, 105.0]
    sd = ta_py.multi.sim(data, l=3, sims=4)
    assert len(sd) == 4
    assert all(len(path) == len(data) + 3 for path in sd)
