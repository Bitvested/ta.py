"""ta.py — Pure-Python technical analysis library."""

from .moving_averages import (
    sma, smma, wma, wsma, pwma, hwma, ema,
    vwma, vwwma, hull, kama, cwma, lsma,
)
from .statistics import (
    median, kmeans, normalize, denormalize, standardize, mad, aad, ssd,
    er, variance, std, normsinv, sim, percentile, cor, dif, drawdown,
    cum, mse, permutations, winratio, avgwin, avgloss, kelly, se,
    zscore, ar, normalize_pair, normalize_from, log, exp, covariance,
    ncdf, pvalue, expected_trails, martingale, antimartingale,
    return_positive, return_negative, recent_high, recent_low, sum,
)
from .indicators import (
    rsi, wrsi, tsi, macd, macd_signal, macd_bars, atr, mfi, roc, mom,
    kst, cop, obv, vwap, ichimoku, fractals, cross, halftrend, zigzag,
    psar, supertrend, elderray, hv, rvi, rvi_signal, rsi_divergence,
    divergence, bop, fi, asi, alligator, pr, stoch, fib,
    aroon_up, aroon_down, aroon_osc,
)
from .oscillators import gator, mom_osc, chaikin_osc, ao, ac, fisher
from .bands import bands, bandwidth, keltner, don, envelope, fibbands
from .chart_types import ha, ren
from .misc import times_up, times_down
from .experimental import support, resistance, divergence_state
from .random_ import prng

__all__ = [
    # moving_averages
    "sma", "smma", "wma", "wsma", "pwma", "hwma", "ema",
    "vwma", "vwwma", "hull", "kama", "cwma", "lsma",
    # statistics
    "median", "kmeans", "normalize", "denormalize", "standardize", "mad", "aad", "ssd",
    "er", "variance", "std", "normsinv", "sim", "percentile", "cor", "dif", "drawdown",
    "cum", "mse", "permutations", "winratio", "avgwin", "avgloss", "kelly", "se",
    "zscore", "ar", "normalize_pair", "normalize_from", "log", "exp", "covariance",
    "ncdf", "pvalue", "expected_trails", "martingale", "antimartingale",
    "return_positive", "return_negative", "recent_high", "recent_low", "sum",
    # indicators
    "rsi", "wrsi", "tsi", "macd", "macd_signal", "macd_bars", "atr", "mfi", "roc", "mom",
    "kst", "cop", "obv", "vwap", "ichimoku", "fractals", "cross", "halftrend", "zigzag",
    "psar", "supertrend", "elderray", "hv", "rvi", "rvi_signal", "rsi_divergence",
    "divergence", "bop", "fi", "asi", "alligator", "pr", "stoch", "fib",
    "aroon_up", "aroon_down", "aroon_osc",
    # oscillators
    "gator", "mom_osc", "chaikin_osc", "ao", "ac", "fisher",
    # bands
    "bands", "bandwidth", "keltner", "don", "envelope", "fibbands",
    # chart types
    "ha", "ren",
    # misc
    "times_up", "times_down",
    # experimental
    "support", "resistance", "divergence_state",
    # random
    "prng",
]

from . import _registry
_registry.registry.update({
    name: globals()[name] for name in __all__
})
