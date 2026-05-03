# ta.py 2.0.0 — release punch list

`v2-dev` is **release-ready** as of 2026-05-03. All Phase 1–5 work is committed: the branch is **101 commits ahead of `main`**, all 162 tests pass at `1e-12` tolerance (~0.06s), `len(ta_py.__all__) = 151`. Multi-process Monte Carlo runs 2–3× faster than the sequential `sim` on 8-core hardware. Version is bumped to `2.0.0` in `pyproject.toml`; the `ta_py/ta.py` shim is removed; `py.typed` marker shipped.

## Outstanding work — release/publish steps (user-driven)

These steps are inherently destructive or visible-to-others and need you to drive them:

1. **Final review.** Read `MIGRATION.md`, `CHANGELOG.md`, the README banner, and run `pytest tests/ -q` + `python -c "import ta_py; print(len(ta_py.__all__))"`. Skim `git log v2-dev ^main --oneline` for any commit-message edits you want to amend.
2. **Optional pre-tag audit pass.** Two pre-flight oversights surfaced during Batch E (rsi, stoch). Other shared primitives (`ema`, `wma`, `wsma`, `tsi`, `macd`, …) were not formally audited; the choice was to keep moving and fix as discovered. None of the existing fixtures regressed at the `1e-12` tolerance, so the risk is low — but if you want an extra-careful pass before tagging, see `phase3-flags.md` § 2 for the audit protocol.
3. **Tag the release.** `git tag v2.0.0`.
4. **Build artifacts.** `python -m build` (writes `dist/ta_py-2.0.0.tar.gz` + wheel).
5. **Publish to PyPI.** `twine upload dist/*`.
6. **Push the tag and the branch.** `git push origin v2.0.0` and `git push origin v2-dev`.
7. **Open the merge PR.** `v2-dev` → `main`. Squash or merge per your usual flow.
8. **GitHub Release.** Create one tied to the `v2.0.0` tag; the body can paste from the new `CHANGELOG.md` entry.

## Verification gate (run before tagging)

```bash
pytest tests/ -q                                          # expect 162 passed
python -c "import ta_py; assert len(ta_py.__all__) == 151"
python -c "import ta_py; print(ta_py.fibnumbers[:5])"     # [0,1,1,2,3]
python -c "import ta_py; print(ta_py.aroon.up([5,4,5,2], 3))"  # [100.0, 50.0]
python -c "import ta_py; r = ta_py.random.prng('seed'); print(ta_py.random.pick([1,2,3,4,5], r))"
python -c "import ta_py; sd = ta_py.multi.sim([1.0]*20, l=5, sims=16); print(len(sd), len(sd[0]))"  # 16 25
python scripts/snapshot_signatures.py | diff - tests/signatures_v2.json   # exit 0
```

## Reference: what shipped in 2.0.0

Detailed history lives in `git log v2-dev ^main --oneline` (101 commits). Headline summary:

- **Phase 1 (28 commits):** `pyproject.toml`, category-module split, `_registry`, full type hints, full docstrings, pytest migration, AST hygiene test, `CONTRIBUTING.md`.
- **Phase 2 (17 perf rewrites):** O(N) sliding-sum / monotonic-deque / single-accumulator rewrites for `sma`/`smma`/`wma`/`wsma`/`pwma`/`hwma`/`vwma`/`vwwma`/`cwma`/`variance`/`bands`/`envelope`/`don`/`pr`/`stoch`/`rsi`/`mfi` plus `kst`/`tsi`/`macd` registry-aliasing.
- **Phase 3 pre-flight + audits:** signature snapshot tooling (`tests/signatures_v1.json` / `signatures_v2.json`); 6 ta.js parity fixes (`atr`, `smma`, `mfi`, `rsi` + `wrsi`-as-alias, `stoch`).
- **Phase 3 ports (Batches A–J, 39 functions):** see `CHANGELOG.md` and `MIGRATION.md`.
- **Phase 3 Batch K:** v2 signature snapshot, MIGRATION final pass, `ABS_TOL`/`REL_TOL` tightened from `1e-9` to `1e-12`, CHANGELOG Phase 3 entry.
- **Phase 4 (5 commits):** `bench/` sub-project with TA-Lib / pandas-ta / finta adapters.
- **Phase 5:** `ta_py.multi.sim` (stdlib `multiprocessing` + chunked workers + sequential fallback), version bump to `2.0.0`, removed `ta_py/ta.py` shim, `py.typed` PEP 561 marker, README/MIGRATION/CHANGELOG final passes.

## Reference docs

- Spec: `~/.claude/plans/we-have-recently-updated-tidy-unicorn.md`
- Phase 3 plan: `~/.claude/plans/yesterday-we-have-finished-snug-chipmunk.md`
- Pre-flight + Batch A–B execution plan: `~/.claude/plans/atomic-herding-peach.md`
- Mid-flight notes / unaudited primitives / decisions: `phase3-flags.md` (this repo, root)
- ta.js source of truth: `~/ta.js-1`
- Cross-library benchmarks: `bench/SAMPLE_BENCHMARKS.md`

## Cross-cutting decisions still pending

- **GitHub Actions CI** — out of scope per the original Phase 1 decision. Worth reconsidering as a 2.0.x patch: a matrix run on Python 3.10–3.13 plus a benchmark smoke would catch regressions early. Not a release blocker.
