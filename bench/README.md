# ta.py benchmark harness

Standalone sub-project. Measures `ta_py` against TA-Lib (C), pandas-ta (numpy),
finta (pandas), and a stdlib-only baseline.

## Install

```bash
cd bench
pip install -e .                       # ta_py + harness only
pip install -e .[talib]                # add TA-Lib
pip install -e .[pandas-ta,finta]      # add pandas competitors
pip install -e .[all]                  # all four competitors
```

TA-Lib needs the system C library installed first (`brew install ta-lib` on macOS).

## Run

```bash
python run.py --sizes 1000,10000,100000 --md --out results/
python run.py --only sma,std,bands --sizes 200 --reference ta_py
```

Outputs `results/BENCHMARKS.md` and `results/results.json`.
