# Technical Analysis (ta.py)

[![PyPI version](https://badge.fury.io/py/ta_py.svg)](https://pypi.org/project/ta_py/) [![Downloads](https://img.shields.io/pypi/dm/ta_py.svg)](https://pypi.org/project/ta_py/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A zero-dependency, pure-Python technical analysis library. **151 indicators**, full type hints, and full functional parity with [ta.js 2.0](https://github.com/Bitvested/ta.js) — its sibling JavaScript library. No NumPy, no pandas, no C extensions, no compilers required.

---

#### NOTE

`ta_py` runs anywhere CPython does. There is nothing to compile and nothing to vendor — `pip install ta_py` is the entire installation.

```python
from ta_py import sma, rsi
sma([1, 2, 3, 4, 5, 6, 10], 6)   # [3.5, 5]
```

The only multi-process feature is `ta_py.multi.sim` (Monte Carlo via stdlib `multiprocessing`); it falls back to the sequential `ta_py.sim` automatically when run inside an environment that can't spawn workers.

## Install

```bash
pip install ta_py
```

Requires Python 3.10 or newer. No other runtime dependencies.

## Usage

```python
import ta_py as ta

ta.sma([1, 2, 3, 4, 5, 6, 10], 6)                              # [3.5, 5]
ta.aroon.up([5, 4, 5, 2], 3)                                   # [100.0, 50.0]
ta.bands([1, 2, 3, 4, 5, 6], length=5, deviations=2)
```

Or import individual functions:

```python
from ta_py import rsi, macd, atr, bands
```

Every function is also re-exported on the top-level `ta_py` module. The `ta_py.aroon`, `ta_py.random`, and `ta_py.multi` namespaces mirror their ta.js counterparts.

## Why pure Python?

`ta_py` is the only TA library on PyPI that combines **zero runtime dependencies**, **full pure-Python implementation**, and **full ta.js 2.0 indicator coverage**. That combination matters when:

- **Install pain is a deal-breaker.** No C compiler, no system libraries, no platform-specific wheels. Works on Alpine containers, AWS Lambda, locked-down corporate Python builds, fresh Windows installs — anywhere `pip` works.
- **Footprint matters.** A few hundred KB on disk vs. ~50–100 MB for `numpy` + `pandas`. Important for serverless cold starts, container images, and edge deploys.
- **You're chasing the latest CPython.** Pure-Python libraries ship the day a new Python release does. C/Cython libraries lag months — `pandas-ta` doesn't yet support Python 3.14, and TA-Lib regularly delays wheels for new interpreters.
- **You can't or won't add native dependencies.** Compliance reviews, supply-chain audits, vendoring requirements: every line of `ta_py` is auditable Python source. No prebuilt binaries, no `setup.py` running compilers.
- **You target alternative runtimes.** PyPy, GraalPy, and Pyodide (Python in the browser via WASM) all run pure-Python code. Most native-extension TA libraries don't port.

If you need raw throughput across millions of bars in tight loops and you're comfortable installing a C extension, use TA-Lib. `ta_py` exists for the much larger case where indicator math is one step in a bigger pipeline and "fast enough on a 10k-bar series, with no install pain" is the right trade.

## Performance notes

Internal sliding-window O(N) rewrites in 2.0 measurably beat naïve list-comprehension implementations of the same indicators (e.g. Bollinger Bands ~5× faster than a textbook Python implementation, standard deviation ~13× faster). The `bench/` sub-project contains a regression-tracking harness used during development; it is **not** a marketing benchmark and does not claim parity with C-backed libraries.

## What's new in 2.0.0

- **39 new indicators** for full ta.js 2.0 parity: DM family (`pdm` / `mdm` / `pdi` / `mdi` / `dx` / `adx` / `adxr` / `natr`), `cci`, `stoch_rsi`, `kdj`, `cmf`, `vortex`, `kvo`, `ult`, `trix`, `ppo`, `apo`, `mass`, `dema` / `tema` / `trima` / `t3` / `zlema` / `vidya`, linear regression family (`lr_slope` / `lr_intercept` / `lr_angle` / `tsf`), `std_series`, `adl`, `nvi` / `pvi`, `emv`, `dpo`, `ulcer`, `fibnumbers`, plus `ta_py.random.*` and `ta_py.aroon.*` namespaces.
- **Multi-process Monte Carlo** via `ta_py.multi.sim` — typically 2–3× faster than the sequential `ta_py.sim` on 8-core hardware.
- **Sliding-window O(N) rewrites** of hot paths: `std`, `bands`, `sma`, `smma`, `wma`, `ema`, `vwma`, `variance`, `envelope`, `mfi`, `rsi`, `stoch`, `pr`, `don`.
- **6 algorithm/output-shape breaks** vs 1.17.0 — `atr`, `smma`, `mfi`, `rsi`, `wrsi` (now an alias of `rsi`), `stoch`. See [MIGRATION.md](MIGRATION.md) for before/after examples.
- **PEP 561 marker** (`py.typed`) ships in the wheel so type-checkers pick up `ta_py`'s annotations automatically.

See [CHANGELOG.md](CHANGELOG.md) for the full per-batch breakdown.

## Examples

Per-indicator usage with input/output samples lives in [**EXAMPLES.md**](EXAMPLES.md).

#### Moving Averages
- [SMA](EXAMPLES.md#sma) · [SMMA](EXAMPLES.md#smma) · [WMA](EXAMPLES.md#wma) · [EMA](EXAMPLES.md#ema) · [DEMA](EXAMPLES.md#dema) · [TEMA](EXAMPLES.md#tema) · [TRIMA](EXAMPLES.md#trima) · [T3](EXAMPLES.md#t3) · [ZLEMA](EXAMPLES.md#zlema) · [VIDYA](EXAMPLES.md#vidya) · [Hull](EXAMPLES.md#hull) · [LSMA](EXAMPLES.md#lsma) · [VWMA](EXAMPLES.md#vwma) · [VWWMA](EXAMPLES.md#vwwma) · [WSMA](EXAMPLES.md#wsma) · [PWMA](EXAMPLES.md#pwma) · [HWMA](EXAMPLES.md#hwma) · [KAMA](EXAMPLES.md#kama) · [CWMA](EXAMPLES.md#cwma)

#### Indicators
- [MACD](EXAMPLES.md#macd) · [MACD signal](EXAMPLES.md#macd_signal) · [MACD bars](EXAMPLES.md#macd_bars) · [RSI](EXAMPLES.md#rsi) · [Wilder's RSI](EXAMPLES.md#wrsi) · [TSI](EXAMPLES.md#tsi) · [BOP](EXAMPLES.md#bop) · [Force Index](EXAMPLES.md#fi) · [ASI](EXAMPLES.md#asi) · [Alligator](EXAMPLES.md#alli) · [Williams %R](EXAMPLES.md#pr) · [Stochastics](EXAMPLES.md#stoch) · [Stoch RSI](EXAMPLES.md#stoch_rsi) · [Fibonacci Retracement](EXAMPLES.md#fib) · [Bollinger Bandwidth](EXAMPLES.md#bandwidth) · [Ichimoku](EXAMPLES.md#ichi) · [ATR](EXAMPLES.md#atr) · [NATR](EXAMPLES.md#natr) · [Aroon Up](EXAMPLES.md#aroon-up) · [Aroon Down](EXAMPLES.md#aroon-down) · [Aroon namespace](EXAMPLES.md#aroon-ns) · [MFI](EXAMPLES.md#mfi) · [CMF](EXAMPLES.md#cmf) · [ROC](EXAMPLES.md#roc) · [Coppock](EXAMPLES.md#cop) · [KST](EXAMPLES.md#kst) · [OBV](EXAMPLES.md#obv) · [NVI](EXAMPLES.md#nvi) · [PVI](EXAMPLES.md#pvi) · [ADL](EXAMPLES.md#adl) · [EMV](EXAMPLES.md#emv) · [VWAP](EXAMPLES.md#vwap) · [Fractals](EXAMPLES.md#fractals) · [Crossover](EXAMPLES.md#cross) · [Momentum](EXAMPLES.md#mom) · [HalfTrend](EXAMPLES.md#half) · [ZigZag](EXAMPLES.md#zigzag) · [PSAR](EXAMPLES.md#psar) · [SuperTrend](EXAMPLES.md#supertrend) · [Elder Ray](EXAMPLES.md#elderray) · [HV](EXAMPLES.md#hv) · [RVI](EXAMPLES.md#rvi) · [RVI signal](EXAMPLES.md#rvi_signal) · [RSI Divergence](EXAMPLES.md#rsi_divergence) · [Universal Divergence](EXAMPLES.md#divergence) · [TRIX](EXAMPLES.md#trix) · [CCI](EXAMPLES.md#cci) · [DM family](EXAMPLES.md#dm) · [DI family](EXAMPLES.md#di) · [DX](EXAMPLES.md#dx) · [ADX](EXAMPLES.md#adx) · [ADXR](EXAMPLES.md#adxr) · [PPO](EXAMPLES.md#ppo) · [APO](EXAMPLES.md#apo) · [DPO](EXAMPLES.md#dpo) · [Mass Index](EXAMPLES.md#mass) · [Ulcer Index](EXAMPLES.md#ulcer) · [Vortex](EXAMPLES.md#vortex) · [KDJ](EXAMPLES.md#kdj) · [KVO](EXAMPLES.md#kvo)

#### Oscillators
- [Alligator Osc](EXAMPLES.md#gator) · [Chande Momentum](EXAMPLES.md#mom_osc) · [Chaikin](EXAMPLES.md#chaikin_osc) · [Aroon Osc](EXAMPLES.md#aroon-osc) · [Awesome](EXAMPLES.md#ao) · [Accelerator](EXAMPLES.md#ac) · [Fisher](EXAMPLES.md#fish) · [Ultimate](EXAMPLES.md#ult)

#### Bands
- [Bollinger](EXAMPLES.md#bands) · [Keltner](EXAMPLES.md#kelt) · [Donchian](EXAMPLES.md#don) · [Fibonacci Bollinger](EXAMPLES.md#fibbands) · [Envelope](EXAMPLES.md#env)

#### Statistics
- [Std](EXAMPLES.md#std) · [Std series](EXAMPLES.md#std_series) · [LR slope](EXAMPLES.md#lr_slope) · [LR intercept](EXAMPLES.md#lr_intercept) · [LR angle](EXAMPLES.md#lr_angle) · [TSF](EXAMPLES.md#tsf) · [Variance](EXAMPLES.md#variance) · [Normal CDF](EXAMPLES.md#ncdf) · [Inverse Normal](EXAMPLES.md#normsinv) · [Monte Carlo](EXAMPLES.md#sim) · [Multi-process Monte Carlo](EXAMPLES.md#multi-sim) · [Percentile](EXAMPLES.md#perc) · [Correlation](EXAMPLES.md#cor) · [Covariance](EXAMPLES.md#cov) · [Pct Difference](EXAMPLES.md#dif) · [Expected Return](EXAMPLES.md#er) · [Abnormal Return](EXAMPLES.md#ar) · [Kelly](EXAMPLES.md#kelly) · [Permutations](EXAMPLES.md#perm) · [Winratio](EXAMPLES.md#winratio) · [Avg Win](EXAMPLES.md#avgwin) · [Avg Loss](EXAMPLES.md#avgloss) · [Drawdown](EXAMPLES.md#drawdown) · [Median](EXAMPLES.md#median) · [Recent High](EXAMPLES.md#rh) · [Recent Low](EXAMPLES.md#rl) · [MAD](EXAMPLES.md#mad) · [AAD](EXAMPLES.md#aad) · [Std Error](EXAMPLES.md#stderr) · [SSD](EXAMPLES.md#ssd) · [Log](EXAMPLES.md#log) · [Exp](EXAMPLES.md#exp) · [Normalize](EXAMPLES.md#norm) · [Denormalize](EXAMPLES.md#dnorm) · [Normalize Pair](EXAMPLES.md#normp) · [Normalize From](EXAMPLES.md#normf) · [Standardize](EXAMPLES.md#standard) · [Z-Score](EXAMPLES.md#zscore) · [K-means](EXAMPLES.md#kmeans) · [MSE](EXAMPLES.md#mse) · [Cumulative](EXAMPLES.md#cum) · [Sum](EXAMPLES.md#sum) · [Return Positive](EXAMPLES.md#return_positive) · [Return Negative](EXAMPLES.md#return_negative) · [P-Value](EXAMPLES.md#pvalue) · [Martingale](EXAMPLES.md#martingale) · [Anti-Martingale](EXAMPLES.md#antimartingale) · [Expected Trails](EXAMPLES.md#expected_trails)

#### Chart Types
- [Heikin Ashi](EXAMPLES.md#ha) · [Renko](EXAMPLES.md#ren)

#### Random
- [Seeded PRNG](EXAMPLES.md#prng)

#### Miscellaneous
- [Fibonacci Sequence](EXAMPLES.md#fibnumbers) · [Times Up](EXAMPLES.md#times_up) · [Times Down](EXAMPLES.md#times_dn)

#### Experimental
- [Support Line](EXAMPLES.md#sup) · [Resistance Line](EXAMPLES.md#res) · [Divergence State](EXAMPLES.md#divergence_state)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Please update tests as appropriate — see [CONTRIBUTING.md](CONTRIBUTING.md) for the test/lint workflow.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Maintained By

**ta.py** is an open-source project created and actively maintained by [**Nino Kroesen**](https://ninokroesen.com).

This library was built with portability in mind for high-frequency data and algorithmic trading systems where install footprint, dependency surface, and supply-chain auditability matter as much as raw throughput. If you'd like to see these indicators in action within a live quantitative trading environment, check out [**Bitvested**](https://bitvested.com).

---
