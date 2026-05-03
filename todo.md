# ta.py 2.0.0 — Outstanding Work

Captured 2026-05-02 at the close of Phase 1; refreshed 2026-05-03 after Phase 1 polish landed and again after Phase 2 + Phase 4 completed. Phase 1 + Phase 2 + Phase 4 are **fully complete** on the `v2-dev` branch (116/116 tests pass). This file lists everything still to do for the 2.0.0 release.

Reference documents:
- Spec: `~/.claude/plans/we-have-recently-updated-tidy-unicorn.md`
- Phase 1 plan: `~/.claude/plans/2026-05-02-phase-1-tooling-restructure.md`
- Phase 1 polish plan: `~/.claude/plans/yesterday-we-have-finished-ticklish-marble.md`
- ta.js reference: `~/ta.js-1`

---

## Phase 1 polish — DONE 2026-05-03

Completed in this order on `v2-dev`:
- `f261424` `_registry.py` PEP 585 + mutation contract docstring.
- `895c02b` Type-hint gaps closed (zigzag, rsi_divergence, halftrend, prng).
- `43794bd` 36 indicator docstrings added (37/37 coverage).
- `1155b4e` 10 inline `from .moving_averages import sma` hoisted to module-top.
- `b71645a` `exp` fixture + parametrized test case (114 -> 115).
- `0f7b112` AST duplicate-def hygiene test (115 -> 116).
- `1b86cba` 9 bare `except:` narrowed; dead `& / .push` block removed from `stoch`.
- `af48e5b` Shim `__all__` rebound to package's canonical list so `math` doesn't leak.
- `3e3041e` `CONTRIBUTING.md` documents the registry-vs-direct convention.

Polish items deliberately deferred:

- **Item 8 — Hot-path `_ema = registry["ema"]` aliasing.** Worth doing alongside the Phase 2 perf rewrites of the same callsites (`kst`, `tsi`, `macd_*`).
- **Item 11 — `inspect.signature` JSON snapshot.** Capture immediately before Phase 3, since Phase 2 may incidentally adjust signatures.

---

## Phase 2 — Performance rewrites of existing hot paths — DONE 2026-05-03

All sliding-window and deque rewrites complete. 116/116 tests pass. Headline speedups at n=10000:
- `std`: 15.98ms -> 1.22ms (~13x)
- `bands`: 21.78ms -> 2.43ms (~9x)
- `sma`: 1.48ms -> 0.58ms (~2.5x — limited by Python interpreter overhead on simple loops)

Completed rewrites:
- O(N) sliding-sum: `sma`, `smma`, `wma`, `wsma`, `vwma`, `variance`, `bands`, `bandwidth`, `keltner`, `envelope`, `mfi`, `rsi`.
- Slice-elimination (still O(N*L) but no per-step list copy): `pwma`, `hwma`, `vwwma`, `cwma`.
- Monotonic-deque: `don`, `pr`, `stoch` (with ring-buffered K/D smoothing).
- Hot-path registry aliasing: `kst`, `tsi`, `macd`, `macd_signal`, `macd_bars`.
- Audited and skipped (already O(N) — no rewrite needed): `recent_high`, `recent_low`, `fractals`.

---

## Phase 3 — Port ~36 new indicators from ta.js 2.0

**Goal:** Functional parity with ta.js 2.0's exported API.

### Moving averages (6 new)

`dema`, `tema`, `trima`, `t3`, `zlema`, `vidya`. Place in `ta_py/moving_averages.py`.

### Indicators (24 new)

`trix`, `adl`, `cci`, `pdm`, `mdm`, `pdi`, `mdi`, `dx`, `adx`, `adxr`, `stoch_rsi`, `ppo`, `apo`, `cmf`, `nvi`, `pvi`, `emv`, `natr`, `dpo`, `mass`, `ulcer`, `vortex`, `kdj`, `elderray` (already there but check signature parity). Place in `ta_py/indicators.py`.

### Oscillators (2 new)

`ult` (Ultimate Oscillator), `kvo` (Klinger Volume Oscillator). Place in `ta_py/oscillators.py`.

### Statistics (5 new)

`lr_slope`, `lr_intercept`, `lr_angle`, `tsf`, `std_series`. Place in `ta_py/statistics.py`.

### Misc (1 new)

`fibnumbers`. Place in `ta_py/misc.py`.

### Random (4 new)

`pick`, `range`, `float`, `order`. Place in `ta_py/random_.py`. Note name clashes -- internally `range_`, `float_` etc.; export via a `random` namespace per the spec.

### Aroon namespace

Convert `aroon_up` / `aroon_down` / `aroon_osc` to a namespace `aroon.up` / `aroon.down` / `aroon.osc` mirroring ta.js. Keep flat names importable for one cycle (deprecation window).

### Signature normalization (breaking)

Diff ta.js v1.17.0 against v2.0.0 signatures and apply the same breaks here:
- `mfi` input tuple shape change (verify against ta.js `test/parameters.js`).
- `atr`, `smma` parameter naming normalizations.
- Default value adjustments where ta.js made them.

Document each break in `MIGRATION.md`.

### Pre-Phase-3 instrumentation (deferred polish item 11)

Snapshot every public function's `inspect.signature(fn)` to a JSON file before Phase 3 starts. Gives a reviewable baseline-diff for the breaking changes. One-time script, ~20 lines.

### Verification

For each new function: copy fixture from `~/ta.js-1/test/parameters.js` into `tests/parameters.py`, then test must pass within `1e-12` of ta.js output. Tighten `ABS_TOL`/`REL_TOL` in `helpers.py` from `1e-9` to `1e-12` once Phase 3 fixtures are authoritative.

### Reference

`~/ta.js-1/src/<category>/<name>.js` for each indicator. Algorithm structures (sliding windows, deques) port directly to Python.

---

## Phase 4 — Cross-library benchmark harness — DONE 2026-05-03 (pulled forward into Phase 2)

`bench/` sub-project fully implemented:
- `bench/harness.py`: stdlib `time.perf_counter_ns` harness with adaptive batch sizing.
- `bench/run.py`: CLI with `--sizes`, `--only`, `--reference`, `--md`, `--json`, `--quick` flags.
- `bench/cases.py`: 17 case definitions covering Phase 2 targets plus EMA + MACD.
- `bench/adapters/`: ta_py + stdlib baseline + optional TA-Lib, pandas-ta, finta wrappers.
- `bench/SAMPLE_BENCHMARKS.md`: committed snapshot of output (ta_py + stdlib; competitors not yet installed).

To run with competitor libs: `pip install -e .[talib,pandas-ta,finta]` then `cd bench && python run.py --sizes 1000,10000 --md`.

---

## Phase 5 — Multi-process Monte Carlo + final polish + 2.0.0 release

### `ta_py.multi.sim`

Port `~/ta.js-1/multi/multi.js` (worker_threads-based parallel Monte Carlo) to Python's stdlib `multiprocessing`. New file `ta_py/multi.py`. Single function `multi.sim(data, sims=10000)` that partitions trials across worker processes and aggregates. No external deps -- `multiprocessing` is stdlib.

Fallback: in PyPy or environments where `multiprocessing` is unavailable, fall back to sequential `sim()`.

Verification: `multi.sim(data)` matches `sim(data)` within statistical tolerance and runs >=2x faster on a 4-core box.

### Final polish

- Bump version to `2.0.0` in `pyproject.toml`.
- Final `MIGRATION.md` pass with the concrete diff between 1.17.0 and 2.0.0 signatures (collected throughout Phase 3).
- Final `README.md` pass: new positioning, function list, link to `BENCHMARKS.md`, `MIGRATION.md` callout for 1.x users.
- Remove the `ta_py/ta.py` deprecation shim. `from ta_py.ta import sma` no longer works after this.
- Tag and release: `git tag v2.0.0`, `python -m build`, `twine upload`.
- Merge `v2-dev` -> `main` only after every step above is green.

### Verification (end-to-end)

1. `pip install -e .` -- installs with zero runtime deps. `pip show ta_py` reports no `Requires:` value.
2. `python -c "import ta_py; print(len(ta_py.__all__))"` -- >=150 (was 112 in Phase 1; +~38 new functions).
3. `pytest tests/` -- all pass with `1e-12` tolerance.
4. `cd bench && python run.py` -- produces `BENCHMARKS.md`.
5. `python -c "from ta_py import multi; print(multi.sim([1,2,3,4,5]*100, sims=1000))"` -- runs without error.
6. `MIGRATION.md` enumerates every breaking change with before/after example.

---

## Cross-cutting decisions still pending

- **Tighten `ABS_TOL` / `REL_TOL`?** Currently `1e-9` in `tests/helpers.py`. Phase 3 fixtures from ta.js are authoritative to `1e-12`. Tighten when those land.
- **GitHub Actions CI?** Out of scope per Phase 1 user decision. Reconsider before 2.0.0 release -- running pytest on Python 3.10-3.13 + a benchmark smoke check would catch regressions early. Could be a Phase 5 add-on.
- **`Typing :: Typed` classifier + `py.typed` marker?** Phase 1 polish item #1 (full docstring coverage) is now done; safe to add the `py.typed` marker any time, signaling the package is type-hint-complete.
- **`[tool.pytest.ini_options]` section in `pyproject.toml`** -- already present per `pytest`'s `configfile: pyproject.toml` line in test runs. Verify on next polish pass.
- **`[project.urls]` expansion** to include `Repository`, `Issues`, `Changelog` once `CHANGELOG.md` lands on `main`.

---

## State at end of Phase 1 + Phase 2 + Phase 4 (for context)

- Branch: `v2-dev` (ahead of `main`); not merged. v2.0.0 must be end-to-end ready before merge.
- Tests: 116/116 pass, ~0.05s.
- Public surface: 112 names in `ta_py.__all__`.
- Backward-compat: `from ta_py.ta import <name>` still works (shim with hardened `__all__`).
- Monkey-patching: verified working via `ta_py._registry.registry["fn"] = ...`.
- Module structure: 9 category modules + `_registry.py` + `ta.py` (20-line shim).
- Indicator coverage: 37/37 docstrings, all type hints in place.
- Docs: `CHANGELOG.md`, `MIGRATION.md`, `CONTRIBUTING.md` present.
- Bench: `bench/` sub-project live; `bench/SAMPLE_BENCHMARKS.md` committed.
- Memory: `~/.claude/projects/-Users-ninokroesen-ta-py/memory/` has user/project/reference/feedback entries that future sessions will load automatically.

When resuming: start by reading the spec at `~/.claude/plans/we-have-recently-updated-tidy-unicorn.md`, then this file. Phase 3 (port ~36 new indicators from ta.js 2.0) is the natural next step.
