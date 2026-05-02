"""ta.py — Pure-Python technical analysis library."""
from .ta import *  # noqa: F401,F403  — to be removed in Task 16 once all functions have moved
from .random_ import prng  # noqa: F401
from .misc import times_up, times_down  # noqa: F401
from .chart_types import ha, ren  # noqa: F401
from .experimental import support, resistance, divergence_state  # noqa: F401
from .bands import bands, bandwidth, keltner, don, envelope, fibbands  # noqa: F401
from .oscillators import gator, mom_osc, chaikin_osc, ao, ac, fisher  # noqa: F401
from .moving_averages import (  # noqa: F401
    sma, smma, wma, wsma, pwma, hwma, ema,
    vwma, vwwma, hull, kama, cwma, lsma,
)
from .statistics import (  # noqa: F401
    median, kmeans, normalize, denormalize, standardize, mad, aad, ssd,
    er, variance, std, normsinv, sim, percentile, cor, dif, drawdown,
    cum, mse, permutations, winratio, avgwin, avgloss, kelly, se,
    zscore, ar, normalize_pair, normalize_from, log, exp, covariance,
    ncdf, pvalue, expected_trails, martingale, antimartingale,
    return_positive, return_negative, recent_high, recent_low, sum,
)
