# Changelog

All notable changes to ta.py are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`occ(data, basis_len=8, basis_type="SMMA", delay=0)`** — Open Close Cross, from JustUncleL's Pine ([source](https://www.tradingview.com/script/IVjPPt6c-Open-Close-Cross-Alert-R6-by-JustUncleL/)), requested in [#4](https://github.com/Bitvested/ta.py/issues/4). Moving averages of `open` and of `close`; the trend is whichever is on top and a flip is the signal. Rows are `[open, close]` and the output aligns to the input bars as `[close basis, open basis, direction]`, NaN until both bases exist. `delay` is the Pine's `delayOffset`, which shifts both bases right.

  Eight of the Pine's twelve `basisType` options are supported: `SMA`, `EMA`, `DEMA`, `TEMA`, `WMA`, `SMMA`, `HullMA`, `TMA`. The other four are omitted rather than approximated — `LSMA` takes an offset parameter `lsma` does not have, `VWMA` needs volume in the data row, and `ALMA` and `SSMA` (Ehlers SuperSmoother) are not implemented in this library. TradingView's alert conditions, alternate-resolution mode and divergence plot are platform features, not computations, and have no counterpart in a library that returns series.

  Two caveats worth knowing. The Pine's variant definitions were checked against the existing primitives for DEMA, TEMA and Hull across even and odd lengths (`~1e-11`, float noise), so those are the same averages. And with an **odd** `basis_len`, Hull's half-length is rounded to even in Python where Pine rounds half away from zero, so that one combination may differ; every other type is unaffected.

### Fixed

- **`alligator` / `gator` no longer match TradingView by accident.** Both were returning their rows newest-first and truncated to `n - jl - js + 2` rows, so they could not be lined up with the input series at all, and each line was displaced by `shift - 1` bars instead of `shift`. `gator`'s upper histogram also had the wrong sign — it plotted `jaw - teeth` instead of `|jaw - teeth|`. Reported in [#10](https://github.com/Bitvested/ta.py/issues/10).

- **`kama` now computes Kaufman's adaptive average correctly.** Four bugs, all inherited from the ta.js original: the smoothing constant dropped its slow term (`- 2/(l3+1) + 2/(l3+1)` cancelled, leaving `ER * fast`); the volatility window summed `l1-1` one-bar changes instead of `l1`, which let the efficiency ratio exceed 1 and divided by zero on flat input; the seed was the SMA of the *last* window rather than the first, so the opening value was computed from data at the end of the series; and the loop started one bar late, emitting `n - l1` values instead of `n - l1 + 1`. Nothing raises before or after — every call simply returns different numbers.

  The recursion now matches tulind, the reference the benchmarks compare against: bit-identical on short, flat, trending and gapping series, max Δ 1.4e-14 on a 100k random walk. Carrying the volatility sum across bars instead of re-summing it per bar also makes it ~2.5× faster. Fixes the same defect as [Bitvested/ta.js#22](https://github.com/Bitvested/ta.js/issues/22), mirrored in ta.go.

- **`halftrend` never flipped, and read its inputs from the wrong bars.** Reported in [#7](https://github.com/Bitvested/ta.py/issues/7). Three defects against the Pine it was ported from ([everget's HalfTrend](https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/)):

  - the flip tests read `sma(...)[0]` — the SMA at the *start* of the window, roughly `atrlen` bars old — instead of the SMA at the current bar. Both flip conditions therefore compared stale values and never fired: on 4901 bars of the reporter's 5-minute data the indicator produced **zero** trend changes and stayed on one line.
  - the price extremes behind `maxlow` / `minhigh` were taken over the ATR window (`atrlen` bars) instead of `amplitude` bars, and the down-flip compared `close` with `<` where the Pine uses `>`.
  - the window was `data[i-atrlen:i]`, excluding the current bar, so every value was one bar stale.

  The rewrite follows the Pine line by line: one row per input bar, extremes over `amplitude`, the state machine evaluated at the current bar, and an ATR that uses Pine's own seeding (Wilder's RMA seeded with the mean of the first `atrlen` true ranges, which is not what `ta_py.atr` does). On the reporter's data it now flips 160 times. Pinned by `tests/test_halftrend.py`, which fails against the previous implementation.

### Changed — BREAKING for `zigzag` consumers

- **`zigzag` now follows the ta.js algorithm**, which the audit showed it did not. Five deviations, all silent: extending a running pivot `continue`d past the bar's move test, delaying every pivot by a bar; the leg reset seeded `min`/`max` with ±infinity where the original seeds them with the new extreme; the flat-branch bounds compared `data[i][0]` but stored `data[i][1]`; and `ZeroDivisionError` fallbacks replaced the original's IEEE arithmetic. On a 300-bar series it disagreed with ta.js and ta.go in 186 of 298 values; it now matches both exactly (max Δ `0.0`), on high/low pairs and flat input alike. `zigzag__pairs` pinned the old interpolation and is re-pinned.

### Changed — BREAKING for `psar` consumers

- **`psar` tracked the wrong extreme in a downtrend.** Where it should have taken the lowest low as the extreme point, it took the high, so the stop-and-reverse sat above the market through a decline and every subsequent value was wrong. On a 300-bar series it disagreed with ta.js and ta.go in 63 values; it now matches both exactly (max Δ `0.0`). `tests/test_indicators.py` pins an 18-bar decline in which the fix moves 12 values.

### Changed — BREAKING for `er`, `ar` and `kelly` consumers

- **A one-sided series is no longer an error.** `er` divided by the count of the side that may be empty, so a series with no losses raised `ZeroDivisionError`, and `ar` and `kelly` raised with it. An empty side carries weight zero, so its geometric mean is never taken and contributes nothing: `er([0.02, 0.01, 0.05, 0.03])` is now `0.0274` (the geometric mean of the wins), `ar` returns finite residuals, and `kelly` returns `1.0` for an all-winning series and `-1.0276` for an all-losing one. Mixed series are unchanged to the last bit.

  The same input produced three different answers across the ports — NaN in ta.js (JS evaluates `1 ** Infinity` as NaN), a finite value in ta.go (where `math.Pow(1, +Inf)` is 1), an exception here — and all three now return these numbers, with the empty side handled explicitly rather than left to each language's exponentiation edge case.

### Changed — BREAKING for `roc`, `kst`, `ha` and `aroon` consumers

Found by a cross-language audit (identical series through 60 indicators in ta.js, ta.go and ta.py): ta.py disagreed with both siblings on these. Nothing raises — every call returns different numbers.

- **`roc` now returns percent, not a fraction.** `100 * (last - reference) / reference`, matching ta.js, ta.go and the usual convention; `ta.roc([1, 2, 3, 4], 3)` was `[2, 1]` and is now `[200.0, 100.0]`. A zero reference yields `NaN` instead of raising `ZeroDivisionError`. `kst` and `cop` are built on `roc` and move with it.
- **`kst`'s positional parameters are reordered** to `(data, r1, r2, r3, r4, s1, s2, s3, s4, sig)` — rates first, then smoothing — which is what ta.js accepts. The old signature interleaved them (`r1, s1, r2, s2, …`), so a positional call that used to mean `r1=5, s1=5, r2=7…` now means `r1=5, r2=5, r3=7…`. The defaults are unchanged in effect, and callers passing keywords are unaffected.
- **`ha` returns `n` rows for `n` inputs**, not `n + 1`. Its loop re-processed bar 0, so the second row was a duplicate computation over the seed. `len(out) == len(data)` now holds, as it does in ta.js and ta.go.
- **`aroon_up` was inverted.** It returned `100 - aroon_up`, i.e. 0 when the window's highest high was its newest bar and 100 when it was the oldest — the opposite of the standard, of ta.js and of ta.go. `aroon_down` was already correct; `aroon_osc` follows `aroon_up`. The pinned `aroon` fixtures never caught it because they placed the extreme at a tie and at the window's middle, the two positions where an inverted Aroon returns the same number; `tests/test_indicators.py` now pins the discriminating cases. A window of one bar has no width, so `aroon_up`/`aroon_down` return `NaN` there instead of 0, matching the siblings.

### Changed — BREAKING for `halftrend` consumers

- **`halftrend` returns one row per input bar** instead of `n - atrlen`, so `len(out) == len(data)` and row `i` covers bar `i`. The ATR channel is `NaN` until the ATR window is full (bar `atrlen - 1`); the trend line itself is defined from bar 0. Callers that index-shifted the old output to line it up with their data — as in the issue report — no longer need to.

### Changed — BREAKING for `alligator` / `gator` consumers

- **`alligator(data, ...)` / `gator(data, ...)` output is now aligned to the input bars.** `len(out) == len(data)` and row `i` covers bar `i`. Each line is displaced right by its own shift, matching TradingView's `plot(..., offset=shift)`, so row `i` holds the SMMA value computed at bar `i - shift`. A line is NaN until bar `length - 1 + shift` (defaults: jaw at 20, teeth at 12, lips at 7). `gator` is now derived from `alligator`'s displaced lines, which also makes it the four-argument-free way to reach TradingView's Gator Oscillator: pass `js=ts=ls=0`.

  Nothing raises — every caller below gets different numbers, not an error:

  - **Length changed** from `n - jl - js + 2` (with defaults, `n - 19`) to `n`. Anything keyed off `len(out)`, sliced by it, or zipped against a same-length series behaves differently.
  - **Row order reversed.** `out[0]` used to be the newest value and `out[-1]` the oldest; now `out[0]` is bar 0 and `out[-1]` is the newest row, which itself holds the value computed 8 bars earlier.
  - **Leading rows are NaN** where they used to be numbers. `sum`, `mean`, `min` or any comparison over the whole array now propagates NaN, and `out[0][0]` is `nan` rather than the latest jaw.
  - **`gator`'s upper histogram changed sign** from `jaw - teeth` (negative) to `|jaw - teeth|` (non-negative). The lower histogram is unchanged.

  The old output was unusable for its stated purpose — that is issue [#10](https://github.com/Bitvested/ta.py/issues/10) — but it was published in 2.0.0, so code written against it will change behaviour silently. See `MIGRATION.md` for the before/after and for how to recover the old array shape if you depended on it.

```python
data = [8, 7, 8, 9, 7, 8, 9, 6, 7, 8, 6, 8, 10, 8, 7, 9, 8, 7, 9, 6, 7, 9]

# 2.0.0 — 3 rows, newest first, not aligned to `data`
ta_py.alligator(data)
# → [[7.726445152480656, 7.813665029359981, 7.6378735707815935],
#    [7.78698224852071, 7.929902890697122, 8.047341963476992],
#    [7.769230769230769, 7.919889017939568, 7.809177454346241]]

# unreleased — 22 rows, one per bar of `data`
ta_py.alligator(data)
# → [[nan, nan, nan], ... 19 more ..., [7.769230769230769, 7.919889017939568, 7.809177454346241],
#    [7.78698224852071, 7.929902890697122, 8.047341963476992]]
```

## [2.0.0] — 2026-05

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
