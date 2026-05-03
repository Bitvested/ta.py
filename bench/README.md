# ta.py internal benchmark harness

This sub-project is the **internal regression-tracking** harness used during ta.py
development. It is **not** a user-facing performance pitch and does **not** claim
parity with C-backed or NumPy-backed libraries — those will always outpace any
pure-Python implementation, and that's by design (see the project README for why
ta.py is pure Python).

What this harness is for:

1. Catching performance regressions across releases on the indicators ta.py ships.
2. Sanity-checking that pure-Python sliding-window rewrites in fact beat their
   naïve textbook counterparts (`stdlib_baseline`).
3. Providing optional cross-checks against TA-Lib / pandas-ta / finta to flag
   numerical drift when an indicator's algorithm changes (see the `Δ` column).

## Install

```bash
cd bench
pip install -e .                       # ta_py + harness only
pip install -e .[talib]                # add TA-Lib (needs system C lib first)
pip install -e .[pandas-ta,finta]      # add pandas competitors
pip install -e .[all]                  # all four optional adapters
```

TA-Lib needs the system C library installed first (`brew install ta-lib` on macOS).

`pandas-ta` requires CPython 3.10–3.13 (no 3.14 wheel as of this writing).

## Run

```bash
python run.py --sizes 1000,10000,100000 --md --out results/
python run.py --only sma,std,bands --sizes 200 --reference ta_py
python run.py --quick                  # 200-bar smoke test of sma/std/bands
```

Outputs `results/BENCHMARKS.md` and `results/results.json`. Both are
git-ignored — regenerate locally on the machine you care about.
