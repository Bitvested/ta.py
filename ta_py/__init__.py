"""ta.py — Pure-Python technical analysis library."""

from .moving_averages import (
    sma, smma, wma, wsma, pwma, hwma, ema,
    vwma, vwwma, hull, kama, cwma, lsma,
    dema, tema, trima, t3, zlema, vidya,
)
from .statistics import (
    median, kmeans, normalize, denormalize, standardize, mad, aad, ssd,
    er, variance, std, normsinv, sim, percentile, cor, dif, drawdown,
    cum, mse, permutations, winratio, avgwin, avgloss, kelly, se,
    zscore, ar, normalize_pair, normalize_from, log, exp, covariance,
    ncdf, pvalue, expected_trails, martingale, antimartingale,
    return_positive, return_negative, recent_high, recent_low, sum,
    lr_slope, lr_intercept, lr_angle, tsf, std_series,
)
from .indicators import (
    rsi, wrsi, tsi, macd, macd_signal, macd_bars, atr, mfi, roc, mom,
    kst, cop, obv, vwap, ichimoku, fractals, cross, halftrend, zigzag,
    psar, supertrend, elderray, hv, rvi, rvi_signal, rsi_divergence,
    divergence, bop, fi, asi, alligator, occ, pr, stoch, fib,
    aroon_up, aroon_down, aroon_osc,
    adl, nvi, pvi, emv, dpo, ulcer,
    pdm, mdm, pdi, mdi, dx, adx, adxr, natr,
    trix, ppo, apo, mass,
    cci, stoch_rsi, kdj,
    cmf, vortex,
)
from .oscillators import gator, mom_osc, chaikin_osc, ao, ac, fisher, kvo, ult
from .bands import bands, bandwidth, keltner, don, envelope, fibbands
from .chart_types import ha, ren
from .misc import times_up, times_down, fibnumbers
from .experimental import support, resistance, divergence_state
from types import SimpleNamespace
from .random_ import prng, pick as _pick, range_ as _range_, float_ as _float_, order as _order

random = SimpleNamespace(prng=prng, pick=_pick, range=_range_, float=_float_, order=_order)
aroon = SimpleNamespace(up=aroon_up, down=aroon_down, osc=aroon_osc)

__all__ = [
    # moving_averages
    "sma", "smma", "wma", "wsma", "pwma", "hwma", "ema",
    "vwma", "vwwma", "hull", "kama", "cwma", "lsma",
    "dema", "tema", "trima", "t3", "zlema", "vidya",
    # statistics
    "median", "kmeans", "normalize", "denormalize", "standardize", "mad", "aad", "ssd",
    "er", "variance", "std", "normsinv", "sim", "percentile", "cor", "dif", "drawdown",
    "cum", "mse", "permutations", "winratio", "avgwin", "avgloss", "kelly", "se",
    "zscore", "ar", "normalize_pair", "normalize_from", "log", "exp", "covariance",
    "ncdf", "pvalue", "expected_trails", "martingale", "antimartingale",
    "return_positive", "return_negative", "recent_high", "recent_low", "sum",
    "lr_slope", "lr_intercept", "lr_angle", "tsf", "std_series",
    # indicators
    "rsi", "wrsi", "tsi", "macd", "macd_signal", "macd_bars", "atr", "mfi", "roc", "mom",
    "kst", "cop", "obv", "vwap", "ichimoku", "fractals", "cross", "halftrend", "zigzag",
    "psar", "supertrend", "elderray", "hv", "rvi", "rvi_signal", "rsi_divergence",
    "divergence", "bop", "fi", "asi", "alligator", "occ", "pr", "stoch", "fib",
    "aroon_up", "aroon_down", "aroon_osc", "aroon",
    "adl", "nvi", "pvi", "emv", "dpo", "ulcer",
    "pdm", "mdm", "pdi", "mdi", "dx", "adx", "adxr", "natr",
    "trix", "ppo", "apo", "mass",
    "cci", "stoch_rsi", "kdj",
    "cmf", "vortex",
    # oscillators
    "gator", "mom_osc", "chaikin_osc", "ao", "ac", "fisher", "kvo", "ult",
    # bands
    "bands", "bandwidth", "keltner", "don", "envelope", "fibbands",
    # chart types
    "ha", "ren",
    # misc
    "times_up", "times_down", "fibnumbers",
    # experimental
    "support", "resistance", "divergence_state",
    # random
    "prng", "random",
]

from . import multi  # noqa: E402,F401 — exposed as ta_py.multi.sim

from . import _registry
_registry.registry.update({
    name: globals()[name] for name in __all__
})
