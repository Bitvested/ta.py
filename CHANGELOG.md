# Changelog

All notable changes to ta.py are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2.0.0

### Added
- `pyproject.toml` (PEP 621) replaces `setup.py` as primary packaging metadata.
- Type hints on every public function.
- One-line docstrings on every public function.
- `ta_py._registry` cross-indicator dispatch dict, enabling runtime monkey-patching of internal calls (e.g., `ta_py._registry.registry["smma"] = my_fn` is picked up by `alligator`, `gator`, etc.).
- pytest-based test suite (`tests/test_*.py`) parameterized from `tests/parameters.py`.

### Performance (Phase 2)

- O(N) sliding-sum rewrites: `sma`, `smma`, `wma`, `vwma`, `variance`, `bands`, `bandwidth` (transitive), `keltner` (transitive), `envelope`, `mfi`, `rsi`.
- O(N) single-accumulator rewrite: `wsma`.
- Slice-elimination rewrites (still O(N·L) but no per-step list slicing): `pwma`, `hwma`, `vwwma`, `cwma`.
- Monotonic-deque rewrites: `don`, `pr`, `stoch` (with ring-buffered K/D smoothing).
- Hot-path registry aliasing in `kst`, `tsi`, `macd`, `macd_signal`, `macd_bars`.
- Headline speedups at n=10000: `std` ~13× (15.98ms → 1.22ms), `bands` ~9× (21.78ms → 2.43ms), `sma` ~2.5× (1.48ms → 0.58ms).
- `recent_high`, `recent_low`, `fractals` audited — already O(N), no rewrite needed.

### Added (Phase 4 pulled forward)

- `bench/` cross-library benchmark sub-project with own `pyproject.toml`.
  - `bench/harness.py`: stdlib `time.perf_counter_ns` harness with adaptive batch sizing.
  - `bench/run.py`: CLI with `--sizes`, `--only`, `--reference`, `--md`, `--json`, `--quick` flags.
  - `bench/cases.py`: 17 case definitions covering Phase 2 perf-rewrite targets plus EMA + MACD.
  - `bench/adapters/`: ta_py + stdlib baseline + optional TA-Lib, pandas-ta, finta wrappers.
  - Optional installs: `pip install -e .[talib]`, `[pandas-ta]`, `[finta]`, or `[all]`.
  - `bench/SAMPLE_BENCHMARKS.md`: committed sample output (ta_py + stdlib only; competitors optional).

### Changed
- `ta_py/ta.py` (1,309-line monolith) split into category modules: `moving_averages.py`, `statistics.py`, `indicators.py`, `oscillators.py`, `bands.py`, `chart_types.py`, `misc.py`, `experimental.py`, `random_.py`. All public symbols remain importable as `from ta_py import <name>`.
- `from ta_py.ta import <name>` continues to work in 1.x via a thin re-export shim (will be removed in 2.0.0 release).

### Fixed
- Removed duplicate `fisher` and `avgwin` definitions. Behavior unchanged: the surviving definitions are the ones that previously won the late-binding race in Python.

## [1.17.0] — 2026-04 (last 1.x release)

- Added: `times_down` function.
- Added: `times_up` function.

## [1.16.0]

- Added: universal divergence.
- Added: RSI Divergence.

## [1.15.0]

- Added: Relative Vigor Index.
- Added: Historical Volatility.

## [Prior versions]

See `git log` for the full history.
