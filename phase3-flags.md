# Phase 3 — Flagged items

Working notes captured during the Batch A–G porting work on `v2-dev`. Things noticed in passing that are NOT immediate blockers but should be reviewed before tagging 2.0.0. Ordered by category, then by urgency.

---

## 1. Algorithm divergences from ta.js 2.0 — RESOLVED in-flight

Both surfaced during Batch E porting and were fixed via separate `fix(indicators)!` commits with `MIGRATION.md` entries. Each was a pre-flight oversight (the formal pre-flight audit covered only `atr`/`smma`/`mfi`).

- **`rsi` — Cutler → Wilder.** ta.py 1.x used a strict sliding window of the last `length-1` deltas (Cutler RSI). ta.js 2.0 uses Wilder smoothing seeded as the mean of the first `length` raw gains/losses. Fixed in commit `b41e038`. `wrsi` is now a thin alias for `rsi` (mirrors ta.js's `wrsi.js`).
- **`stoch` — extended window → length-only.** ta.py 1.x used `l1 + sk + sd - 1` as the rolling-max/min window (preserved through Phase 2). ta.js 2.0 uses just `length`, then layers the `sk` and `sd` SMAs separately. Fixed in commit `fa8a823`. Function now also auto-swaps `length`/`sd` and clamps `sk ≤ sd` to mirror ta.js's lenient handling. Output gains `sk + sd - 2` rows.

---

## 2. Pre-flight gaps NOT YET audited (probable risks)

Phase 3 keeps surfacing algorithm divergences during porting because the formal pre-flight only covered `atr`/`smma`/`mfi`. Each batch that chains into a shared ta.py primitive carries the risk of another quiet break. The audit pattern is: pick the ta.py primitive, run the same input through both `ta_py.X(...)` and `node -e "console.log(require('./index.js').X(...))"` from `~/ta.js-1`, compare values **and** output length.

Primitives that future Phase 3 batches (and Phase 5 verification) chain into, ranked by exposure:

| Primitive | Used by | Status |
|-----------|---------|--------|
| `ema` | trix, ppo, mass, kvo, tsi, macd, alligator, gator, kama, dema, tema, t3, zlema, vidya, … | **Unaudited.** ta.py uses simple SMA seed + classic recurrence. Worth a direct check against ta.js since EMA seed conventions vary across libraries. |
| `wma` | hull, cop, vwwma | **Unaudited.** Phase 2 rewrite preserved 1.x behavior; need to confirm 1.x matched ta.js. |
| `wsma` | wrsi (now alias-only), envelope path? | **Unaudited.** Same situation as wma. |
| `macd` / `macd_signal` / `macd_bars` | ppo, apo, kst | **Unaudited.** Fixtures pass but they were copied forward from ta.py 1.x — not from ta.js 2.0. |
| `tsi` | (standalone) | **Unaudited.** Existing fixture matches ta.py 1.x; not cross-checked against ta.js 2.0. |
| `mad` (statistics) | cci (we just inlined the rolling MAD; doesn't reuse `mad`) | Indirectly fine for Batch E, but if Batch H reuses, audit. |
| `sma` | many | Already implicitly validated through dpo/kst/ppo passing, but worth a length check. |

**Recommendation:** before Batch K (the verification + tolerance step), do a focused audit pass: for each name in this table, run a small node fixture and compare. Any mismatch becomes a `fix!` commit with a `MIGRATION.md` entry.

---

## 3. Open algorithmic questions

- **`std_series` vs existing rolling `std`** (Batch H). The Phase 3 plan flagged this directly: "Confirm `std_series` is distinct from existing rolling `std(data, l1=N)` in ta.py's `statistics.py`." The current `std` fixture (`{"in": ([1, 2, 3], 3), "out": 0.8164965809277261}`) returns a SCALAR — it's the sample stdev of the whole series. ta.js 2.0's `std_series` returns a rolling list. So they're distinct. The two should coexist; Batch H should add `std_series` without touching the existing scalar `std`. Flag for Batch H execution time: do not rename or replace existing `std`.

- **`kdj` parameter ordering vs ta.js's `stoch`** (closed in Batch E). Worth recording: ta.js's `kdj.js` calls `ta.stoch(data, length, smoothK, smoothD)` while ta.js's `stoch` signature is `(data, length, smoothd, smoothk)`. Effectively kdj passes its `smoothK` value into `stoch`'s `smoothd` slot and vice-versa. ta.js's `stoch` then auto-swaps if `sk > sd`, masking this mismatch for typical inputs. ta.py kdj mirrors the bug-compatible behavior. If anyone audits ta.js source directly and is confused — this is why.

---

## 4. Plan-vs-reality discrepancies in `~/.claude/plans/yesterday-we-have-finished-snug-chipmunk.md`

Minor inaccuracies that don't block work but would mislead a future session reading the plan:

- **Batch B fixture line for `ulcer`.** Plan cites `parameters.js:284`. Actual `parameters.js:284` is the `dpo` entry; `ulcer` is at the next entry (`parameters.js:288`). Resolved during port.
- **Batch C input format note.** Plan says "ta.js fixtures use `[H, L, C]` triplets — confirm ta.py keeps the same input shape; do NOT silently switch to `[H, C, L]`". The actual ta.js DM-family source uses `[H, C, L]` (close at index 1, low at index 2) — same as ta.py atr. The plan note had the order wrong. Resolved during port.
- **Batch E note: "kdj has no ta.js fixture — capture by running `node`".** Actually `parameters.js` *does* contain a `kdj` fixture. Resolved during port.

---

## 5. Cross-cutting decisions still pending (from `todo.md`)

These are unchanged from the start of Phase 3; carrying forward so they don't get lost:

- **Tighten `ABS_TOL` / `REL_TOL`?** Currently `1e-9` in `tests/helpers.py`. Phase 3 fixtures from ta.js are authoritative to `1e-12`. Tighten in Batch K.
- **GitHub Actions CI?** Out of scope per Phase 1 user decision. Reconsider before 2.0.0 release — running pytest on Python 3.10–3.13 + a benchmark smoke check would catch regressions early. Could be a Phase 5 add-on.
- **`Typing :: Typed` classifier + `py.typed` marker?** Phase 1 polish item #1 (full docstring coverage) is now done; safe to add the `py.typed` marker any time.
- **`[tool.pytest.ini_options]` section in `pyproject.toml`** — already present per `pytest`'s `configfile: pyproject.toml` line in test runs. Verify on next polish pass.
- **`[project.urls]` expansion** to include `Repository`, `Issues`, `Changelog` once `CHANGELOG.md` lands on `main`.

---

## 6. Process notes for future sessions

- **Per-indicator commit cadence works.** Each of Batches B–G landed as N small commits. Bisectable, easy to revert. Continue this through Batches H–J.
- **`todo.md` updates kept uncommitted between batches per user direction.** When the user is ready to commit, all the staged updates can fold into one `docs(todo)` commit at end of Phase 3.
- **`scripts/snapshot_signatures.py` already handles non-callables** (e.g., `fibnumbers` constant) via the `kind: "value"` branch. When Batch K re-runs the snapshot for `tests/signatures_v2.json`, no script change needed.
- **Pre-flight pattern that worked.** When a divergence is found mid-batch, the cleanest sequence is: (1) reset the in-progress port to HEAD, (2) commit the alignment as `fix(indicators)!` with MIGRATION entry, (3) re-apply the port and commit. Used twice in Batch E (rsi + stoch); applies cleanly to per-indicator workflows.
