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

- `bench/` internal regression-tracking harness (separate sub-project with its own `pyproject.toml`).
  - `bench/harness.py`: stdlib `time.perf_counter_ns` timer with adaptive batch sizing.
  - `bench/run.py`: CLI with `--sizes`, `--only`, `--reference`, `--md`, `--json`, `--quick` flags.
  - `bench/cases.py`: 17 case definitions covering Phase 2 perf-rewrite targets plus EMA + MACD.
  - `bench/adapters/`: ta_py adapter + naïve stdlib baseline + optional TA-Lib / pandas-ta / finta wrappers (used as cross-checks during development; not a marketing comparison — C-backed and numpy-backed libraries are expected to outpace any pure-Python implementation).
  - Optional installs: `pip install -e .[talib]`, `[pandas-ta]`, `[finta]`, or `[all]`.

### Changed
- `ta_py/ta.py` (1,309-line monolith) split into category modules: `moving_averages.py`, `statistics.py`, `indicators.py`, `oscillators.py`, `bands.py`, `chart_types.py`, `misc.py`, `experimental.py`, `random_.py`. All public symbols remain importable as `from ta_py import <name>`.
- `from ta_py.ta import <name>` continues to work in 1.x via a thin re-export shim (will be removed in 2.0.0 release).

### Fixed
- Removed duplicate `fisher` and `avgwin` definitions. Behavior unchanged: the surviving definitions are the ones that previously won the late-binding race in Python.

### Added (Phase 3 — ta.js 2.0 parity)

- **39 new public names** taking `len(ta_py.__all__)` from 112 to 151. Full functional parity with ta.js 2.0.
- **Moving averages (Batch A, 6):** `dema`, `tema`, `trima`, `t3`, `zlema`, `vidya`.
- **Standalone indicators (Batch B, 6 + constant):** `adl`, `nvi`, `pvi`, `emv`, `dpo`, `ulcer`; plus `fibnumbers` constant array.
- **Directional Movement family (Batch C, 8):** `pdm`, `mdm`, `pdi`, `mdi`, `dx`, `adx`, `adxr`, `natr`.
- **Multi-EMA derivatives (Batch D, 4):** `trix`, `ppo`, `apo`, `mass`.
- **Stochastic / typical-price family (Batch E, 3):** `cci`, `stoch_rsi`, `kdj`.
- **Volume oscillators (Batch F, 3):** `cmf`, `vortex`, `kvo`.
- **Ultimate Oscillator (Batch G, 1):** `ult`.
- **Linear regression family + rolling stdev (Batch H, 5):** `lr_slope`, `lr_intercept`, `lr_angle`, `tsf`, `std_series` via shared `_lr_windows` helper.
- **Random namespace (Batch I, 4 helpers + namespace):** `pick`, `range`, `float`, `order` exposed via `ta_py.random.*` alongside existing `prng`.
- **Aroon namespace (Batch J):** `ta_py.aroon.up` / `.down` / `.osc` SimpleNamespace; flat `aroon_up` / `aroon_down` / `aroon_osc` continue to work as silent aliases.

### Changed (Phase 3 — algorithm/output-shape breaks vs 1.17.0)

- **`atr(data, length)`** — now uses absolute values for `H − Cprev` and `L − Cprev` (Wilder's original definition); output trimmed to `n − length + 1` rows. Cascades to `keltner`, `supertrend`, `halftrend`.
- **`smma(data, length)`** — switched to standard Wilder smoothing seeded from SMA. Output now `n − length + 1` rows. Cascades to `alligator`, `gator`.
- **`mfi(data, length)`** — input changed from precomputed `[positive_flow, negative_flow]` pairs to raw OHLCV `[H, L, C, V]` rows. Output is `n − length`.
- **`rsi(data, length)`** — switched from Cutler RSI (sliding window of deltas) to Wilder smoothing (matches ta.js 2.0 / TA-Lib). Output is `n − length`.
- **`wrsi(data, length)`** — now a thin alias for `rsi` (mirrors ta.js 2.0's `wrsi.js`).
- **`stoch(data, length, sd, sk)`** — rolling-max/min window changed from extended `length + sk + sd − 1` bars to `length` bars; smoothing layered separately. Output gains `sk + sd − 2` rows. Function also auto-swaps `length`/`sd` and clamps `sk ≤ sd` for ta.js parity.

All six are documented in `MIGRATION.md` with before/after examples.

### Tooling (Phase 3)

- `scripts/snapshot_signatures.py` + `tests/signatures_v1.json` (1.17.0 baseline) + `tests/signatures_v2.json` (2.0.0 surface) capture and diff the public API. Net: 39 added, 0 removed, 0 signature changes.
- Float tolerance tightened from `1e-9` to `1e-12` in `tests/helpers.py`.

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
