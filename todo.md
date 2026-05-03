# ta.py 2.0.0 — Outstanding Work

Captured 2026-05-02 at the close of Phase 1; refreshed 2026-05-03 after Phase 1 polish, Phase 2 + Phase 4, and the Phase 3 pre-flight + Batch A. Phase 1 + Phase 2 + Phase 4 are **fully complete** on the `v2-dev` branch. Phase 3 is **in progress**: pre-flight (signature snapshot + atr/smma/mfi audit) and Batch A (6 moving averages) are landed; Batches B–K outstanding. Tests: 122/122 pass at `1e-9`.

Reference documents:
- Spec: `~/.claude/plans/we-have-recently-updated-tidy-unicorn.md`
- Phase 1 plan: `~/.claude/plans/2026-05-02-phase-1-tooling-restructure.md`
- Phase 1 polish plan: `~/.claude/plans/yesterday-we-have-finished-ticklish-marble.md`
- Phase 2 + Phase 4 plan: `~/.claude/plans/elegant-gliding-lightning.md`
- Phase 3 plan: `~/.claude/plans/yesterday-we-have-finished-snug-chipmunk.md`
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

Polish items previously deferred:

- **Item 8 — Hot-path `_ema = registry["ema"]` aliasing.** DONE in Phase 2 (commit `823c87e`). Aliased registry lookups in `kst` (4 in-loop callsites), `tsi`, `macd`, `macd_signal`, `macd_bars`. Note: the deferred-list entry mentioned `macd_hist` but the actual function name is `macd_bars`.
- **Item 11 — `inspect.signature` JSON snapshot.** DONE 2026-05-03 as the Phase 3 pre-flight. `scripts/snapshot_signatures.py` captures every name in `ta_py.__all__` to JSON; v1 baseline lives at `tests/signatures_v1.json` (112 entries). Re-run at the end of Phase 3 to produce `tests/signatures_v2.json` for the MIGRATION.md diff.

---

## Phase 2 — Performance rewrites of existing hot paths — DONE 2026-05-03

All sliding-window and deque rewrites complete. 116/116 tests pass at `1e-9`. Public surface unchanged. Headline speedups at n=10000:
- `std`: 15.98ms -> 1.22ms (~13x) ✅ ≥10× gate
- `bands`: 21.78ms -> 2.43ms (~9x) ✅ within ≥8× target
- `sma`: 1.48ms -> 0.58ms (~2.5x — limited by Python interpreter overhead on simple loops)

Completed in this order on `v2-dev`:
- `b56a37a` `sma` O(N) sliding sum.
- `25ff029` `smma` O(N) sliding sum (preserves seed-discard quirk).
- `3cbf8a3` `wma` O(N) running weighted sum.
- `d5e7b30` `wsma` O(N) single accumulator.
- `0110155` `pwma` slice elimination.
- `0773a1a` `hwma` slice elimination.
- `d0577b8` `vwma` O(N) dual sliding sum.
- `ddfc996` `vwwma` slice elimination.
- `d24912e` `cwma` slice elimination.
- `28f7178` `variance` O(N) dual-sum (E[X²] - E[X]²) — `std` inherits 13× speedup.
- `07e1a3b` `bands` O(N) dual-sum — `bandwidth` and `keltner` inherit.
- `9e8b791` `envelope` O(N) sliding sum.
- `19e9bee` `don` O(N) monotonic deque.
- `38dff64` `pr` O(N) monotonic deque.
- `ea9fe40` `stoch` O(N) deque + ring-buffer K/D smoothing (uses effective window `l1+sk+sd-1` to preserve original output).
- `65b65cc` `rsi` O(N) sliding gain/loss accumulators.
- `f0d8822` `mfi` O(N) dual sliding sum.
- `823c87e` Item 8 — hot-path registry aliasing in `kst`/`tsi`/`macd`/`macd_signal`/`macd_bars`.

Audited and skipped (already O(N) — no rewrite needed): `recent_high`, `recent_low`, `fractals`. The `todo.md` previously listed these under deque rewrites; they don't actually need it.

---

## Phase 3 — Port ta.js 2.0 indicators + signature normalization

**Goal:** Functional parity with ta.js 2.0's exported API. Plan: `~/.claude/plans/yesterday-we-have-finished-snug-chipmunk.md`. Tracked as 11 batches (Pre-flight + A–K). Open scope decisions are recorded in the plan: silent aroon aliases through 2.x, tighten tolerance to `1e-12` at the end of Phase 3.

### Phase 3 — Pre-flight — DONE 2026-05-03

**Signature snapshot (deferred polish item 11) — DONE.**
- `scripts/snapshot_signatures.py` (~30 LOC) walks `ta_py.__all__` and records each `inspect.signature` to JSON.
- `tests/signatures_v1.json` committed as the v1.x baseline (112 entries).
- Re-run at end of Phase 3 → `tests/signatures_v2.json` → diff into `MIGRATION.md`.

**`atr` / `smma` / `mfi` audit vs ta.js 2.0 — DONE.** All three needed real fixes (none were "no change required"):
- `atr` (`indicators.py`): added `abs()` to the H−Cprev and L−Cprev terms; output dropped to `n − length + 1` rows. Fix cascaded to `supertrend` (re-indexed for new atr shape) and `halftrend` (loop tightened from `range(atrlen, n+1)` → `range(atrlen, n)` to match ta.js, deviation arms shifted accordingly).
- `smma` (`moving_averages.py`): replaced an idiosyncratic `(sliding_sum − prev) / l` recurrence with standard Wilder smoothing seeded from SMA (`prev = prev + (data[i] − prev) / l`); output now `n − length + 1` rows. Fix cascaded to `alligator` and `gator` (output values shift; existing slice-trim logic continues to align them).
- `mfi` (`indicators.py`): input format changed from precomputed `[positive_flow, negative_flow]` pairs to raw OHLCV `[H, L, C, V]` rows. Internal pos/neg flow tracking ported from ta.js 2.0 mfiVal helper.
- Fixtures regenerated for: `atr`, `smma`, `mfi`, `keltner`, `alligator`, `gator`, `supertrend`, `halftrend`. All against ta.js 2.0 reference values.
- Each break documented in `MIGRATION.md` with before/after example.

### Phase 3 — Batch A (moving averages) — DONE 2026-05-03

Six new functions in `ta_py/moving_averages.py`:
- `dema(data, length=30)` — streaming dual-EMA, output `n − 2*length + 2` rows.
- `tema(data, length=30)` — `3*EMA1 − 3*EMA2 + EMA3` via three chained `ema` calls.
- `trima(data, length=30)` — SMA of SMA with split lengths.
- `t3(data, length=5, vfactor=0.7)` — six chained EMAs with polynomial blend.
- `zlema(data, length=14)` — streaming zero-lag EMA, output length matches input.
- `vidya(data, short_length=2, long_length=5, alpha=0.2)` — volatility-adaptive EMA via `registry["std"]` ratio.

Six fixtures added to `tests/parameters.py` (verbatim from ta.js `test/parameters.js`). `tests/test_moving_averages.py` `CASES` extended. `__init__.py` imports + `__all__` updated. `len(ta_py.__all__)` = 118 (was 112). Tests: 122/122 (was 116) pass at `1e-9`.

### Phase 3 — Batches B–K outstanding

The plan file (`~/.claude/plans/yesterday-we-have-finished-snug-chipmunk.md`) is the authoritative breakdown. Summary:

- **Batch B — Standalone indicators (7).** `adl`, `nvi`, `pvi`, `emv`, `dpo`, `ulcer` in `indicators.py`; `fibnumbers` constant in `misc.py`.
- **Batch C — DM family (8).** `pdm` → `mdm` → `pdi` → `mdi` → `dx` → `adx` → `adxr` → `natr` in `indicators.py`. Port in dependency order.
- **Batch D — Multi-EMA derivatives (4).** `trix`, `ppo`, `apo`, `mass` in `indicators.py`.
- **Batch E — Stoch / typical-price family (3).** `cci`, `stoch_rsi`, `kdj` in `indicators.py`. (Note: `kdj` has no ta.js fixture — capture by running `node` against `~/ta.js-1`.)
- **Batch F — Volume oscillators (3).** `cmf`, `vortex` in `indicators.py`; `kvo` in `oscillators.py`.
- **Batch G — Ultimate Oscillator (1).** `ult` in `oscillators.py`.
- **Batch H — Linear regression family (5).** `lr_slope`, `lr_intercept`, `lr_angle`, `tsf`, `std_series` in `statistics.py` via shared `_lr_windows` helper. Confirm `std_series` is distinct from existing rolling `std` before duplicating.
- **Batch I — Random namespace (4).** `pick`, `range_`, `float_`, `order` in `random_.py`; `ta_py.random` SimpleNamespace exposes them as `pick`/`range`/`float`/`order` alongside existing `prng`. Don't flat-export the four — they shadow builtins under `from ta_py import *`.
- **Batch J — Aroon namespace.** `ta_py.aroon` SimpleNamespace exposing `up`/`down`/`osc`. Flat `aroon_up` / `aroon_down` / `aroon_osc` stay as silent aliases through 2.x (user-confirmed: no DeprecationWarning).
- **Batch K — Verification + tolerance + docs.** Re-snapshot signatures → `tests/signatures_v2.json` → diff into `MIGRATION.md`. Tighten `ABS_TOL` / `REL_TOL` in `tests/helpers.py` from `1e-9` to `1e-12`. Confirm `len(ta_py.__all__) ≥ 150`. Update `CHANGELOG.md`.

### Phase 3 — Reference

`~/ta.js-1/src/<category>/<name>.js` for each indicator. Algorithm structures (sliding windows, deques) port directly to Python. Reuse Phase 2 patterns: sliding sum / dual-sum (`bands.py:5–32`), monotonic deque rolling max (`don`, `pr`), registry-aliasing in hot loops (`kst`).

---

## Phase 4 — Cross-library benchmark harness — DONE 2026-05-03 (pulled forward into Phase 2)

`bench/` sub-project fully implemented in 5 commits:
- `979d9cc` Skeleton + `pyproject.toml` (own project with `[talib]`, `[pandas-ta]`, `[finta]`, `[all]` extras).
- `56a2bed` `bench/harness.py`: stdlib `time.perf_counter_ns` harness with adaptive batch sizing.
- `233a4d0` `bench/adapters/` for ta_py + stdlib baseline + optional TA-Lib / pandas-ta / finta wrappers (skip-if-import-fails pattern).
- `da28f6d` `bench/cases.py`: 17 case definitions covering Phase 2 targets plus `ema` + `macd`.
- `08c41a7` `bench/run.py`: CLI with `--sizes`, `--only`, `--reference`, `--md`, `--json`, `--quick` flags.
- `fbd412e` `bench/SAMPLE_BENCHMARKS.md`: committed snapshot of output (ta_py + stdlib; competitors not yet installed).

To run with competitor libs: `cd bench && pip install -e .[all]` then `python run.py --sizes 1000,10000 --md`. Outputs `bench/results/BENCHMARKS.md` (gitignored).

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

## State at end of Phase 3 pre-flight + Batch A (for context)

- Branch: `v2-dev` — **53 commits ahead of `main`** at session start; pre-flight + Batch A work is staged but **uncommitted** as of 2026-05-03 close. v2.0.0 must be end-to-end ready before merge.
- Tests: 122/122 pass at `1e-9` tolerance (~0.05s) — was 116/116 at start of session.
- Public surface: 118 names in `ta_py.__all__` (was 112; Batch A added 6).
- Signature snapshot: `tests/signatures_v1.json` is the v1.x baseline (112 entries, captured before any Phase 3 breaks). Re-snapshot at end of Phase 3 → `tests/signatures_v2.json` for the migration diff.
- Breaking changes already landed in pre-flight: `atr` (abs-fix + n−l+1 output), `smma` (Wilder recurrence + n−l+1 output), `mfi` (OHLCV input format). Cascaded fixture updates: `keltner`, `alligator`, `gator`, `supertrend`, `halftrend`. All documented in `MIGRATION.md`.
- Backward-compat: `from ta_py.ta import <name>` still works (shim with hardened `__all__`).
- Monkey-patching: verified working via `ta_py._registry.registry["fn"] = ...`.
- Module structure: 9 category modules + `_registry.py` + `ta.py` (20-line shim) + new `scripts/` directory.
- Indicator coverage: 37/37 docstrings, all type hints in place. Six new MAs (dema/tema/trima/t3/zlema/vidya) docstring + type-hint complete.
- Docs: `CHANGELOG.md` (Phase 2 + Phase 4 entries added; needs Phase 3 entry at Batch K), `MIGRATION.md` (atr/smma/mfi sections added), `CONTRIBUTING.md` present.
- Bench: `bench/` sub-project live (5 commits + sample). `bench/SAMPLE_BENCHMARKS.md` committed; `bench/results/` gitignored.
- Memory: `~/.claude/projects/-Users-ninokroesen-ta-py/memory/` has user/project/reference/feedback entries that future sessions will load automatically.

When resuming Phase 3: start by reading the Phase 3 plan at `~/.claude/plans/yesterday-we-have-finished-snug-chipmunk.md`, then this file. Pick up at Batch B (standalone indicators: adl/nvi/pvi/emv/dpo/ulcer + fibnumbers constant). The plan groups remaining batches by dependency chain so each batch is bisectable. Same per-batch cadence as Phases 1/2: read ta.js source → port → fixture from ta.js → extend `CASES` → pytest green → commit.
