"""Oscillator indicators."""
import math
from ._registry import registry


def gator(
    data: list[float],
    jl: int = 13, tl: int = 8, ll: int = 5,
    js: int = 8, ts: int = 5, ls: int = 3,
) -> list[list[float]]:
    """Gator Oscillator (alligator-derived).

    ``[|jaw - teeth|, -|teeth - lips|]`` per bar, taken from the displaced
    Alligator lines, so ``len(out) == len(data)`` and row ``i`` covers bar ``i``.
    TradingView's Gator Oscillator does not displace its lines: pass
    ``js=ts=ls=0`` to match it.
    """
    return [
        [abs(jaw - teeth), -abs(teeth - lips)]
        for jaw, teeth, lips in registry["alligator"](data, jl, tl, ll, js, ts, ls)
    ]


def mom_osc(data: list[float], l1: int = 9) -> list[float]:
    """Chande Momentum Oscillator."""
    pl = []; osc = []; l1+=1;
    for i in range(len(data)):
        pl.append(data[i]);
        if(len(pl) >= l1):
            sumh = 0; suml = 0;
            for a in range(1, l1):
                if(pl[a-1] < pl[a]): sumh = sumh+pl[a];
                else: suml = suml+pl[a];
            osc.append((sumh - suml) / (sumh + suml) * 100);
            pl = pl[1:];
    return osc;


def chaikin_osc(data: list[list[float]], ema1: int = 3, ema2: int = 10) -> list[float]:
    """Chaikin Oscillator. Each row of data is [high, low, close, volume]."""
    cha = []; adl = [];
    for i in range(len(data)):
        mfm = ((data[i][1]-data[i][2])-(data[i][0]-data[i][1]))/(data[i][0]-data[i][2]) if data[i][0]-data[i][2] != 0 else 0;
        adl.append(mfm*data[i][3]);
    ef = registry["ema"](adl, ema1); es = registry["ema"](adl, ema2);
    if len(ef) > len(es):
        ef = ef[len(ef)-len(es):];
    else:
        es = es[len(es)-len(ef):];
    for i in range(len(ef)):
        cha.append(ef[i]-es[i]);
    return cha;


def ao(data: list[list[float]], l1: int = 5, l2: int = 35) -> list[float]:
    """Awesome Oscillator. Each row of data is [high, low]."""
    pl = []; a = [];
    for i in range(len(data)):
        pl.append((float(data[i][0]) + float(data[i][1])) / 2.0);
        if(len(pl) >= l2):
            f = registry["sma"](pl[:], l1);
            s = registry["sma"](pl[:], l2);
            a.append(f[len(f)-1] - s[len(s)-1]);
            pl = pl[1:];
    return a;


def ac(data: list[list[float]], l1: int = 5, l2: int = 35) -> list[float]:
    """Accelerator Oscillator. Each row of data is [high, low]. Calls ao() within same module."""
    pl = [];
    for i in range(len(data)):
        pl.append((data[i][0]+data[i][1])/2);
    a = ao(data, l1, l2); sm = registry["sma"](a, l1); acr = [];
    if len(a) > len(sm):
        a = a[len(a)-len(sm):]
    else:
        sm = sm[len(sm)-len(a):];
    for i in range(len(a)): acr.append(a[i]-sm[i]);
    return acr;


def fisher(data: list[float], l: int) -> list[list[float]]:
    """Fisher Transform."""
    out = []; fish = 0; v1 = 0;
    for i in range(l, len(data)+1):
        pl = data[i-l:i]; pf = fish;
        mn = min(pl);
        v1 = .33*2*((data[i-1]-mn)/(max(pl)-mn)-.5)+.67*v1;
        if v1 > .99:
            v1 = .999;
        if v1 < -.99:
            v1 = -.999;
        fish = 0.5 * math.log((1+v1)/(1-v1)) + 0.5 * pf;
        out.append([fish, pf]);
    return out[1:];


def ult(data: list[list[float]], p1: int = 7, p2: int = 14, p3: int = 28) -> list[float]:
    """Williams Ultimate Oscillator. Each row of data is [high, close, low].

    For each bar i>=1: BP = C - min(L, Cprev), TR = max(H, Cprev) - min(L, Cprev).
    Avg_n = ΣBP / ΣTR over the trailing ``n`` bars; ULT = 100·(4·Avg_p1 +
    2·Avg_p2 + Avg_p3) / 7. Output length is ``n - max(p1, p2, p3)``.
    """
    n = len(data)
    need = max(p1, p2, p3)
    if n - 1 < need:
        return []
    bp = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        h = float(data[i][0]); c = float(data[i][1]); lo = float(data[i][2])
        cprev = float(data[i - 1][1])
        low_v = min(lo, cprev)
        bp[i] = c - low_v
        tr[i] = max(h, cprev) - low_v
    s_bp1 = 0.0; s_tr1 = 0.0; s_bp2 = 0.0; s_tr2 = 0.0; s_bp3 = 0.0; s_tr3 = 0.0
    for i in range(need - p1 + 1, need + 1):
        s_bp1 += bp[i]; s_tr1 += tr[i]
    for i in range(need - p2 + 1, need + 1):
        s_bp2 += bp[i]; s_tr2 += tr[i]
    for i in range(need - p3 + 1, need + 1):
        s_bp3 += bp[i]; s_tr3 += tr[i]
    def _ult_val(b1: float, t1: float, b2: float, t2: float, b3: float, t3: float) -> float:
        a1 = float("nan") if t1 == 0 else b1 / t1
        a2 = float("nan") if t2 == 0 else b2 / t2
        a3 = float("nan") if t3 == 0 else b3 / t3
        return 100.0 * (4.0 * a1 + 2.0 * a2 + a3) / 7.0
    out: list[float] = [_ult_val(s_bp1, s_tr1, s_bp2, s_tr2, s_bp3, s_tr3)]
    for i in range(need + 1, n):
        s_bp1 += bp[i] - bp[i - p1]; s_tr1 += tr[i] - tr[i - p1]
        s_bp2 += bp[i] - bp[i - p2]; s_tr2 += tr[i] - tr[i - p2]
        s_bp3 += bp[i] - bp[i - p3]; s_tr3 += tr[i] - tr[i - p3]
        out.append(_ult_val(s_bp1, s_tr1, s_bp2, s_tr2, s_bp3, s_tr3))
    return out


def kvo(data: list[list[float]], fast: int = 34, slow: int = 55) -> list[float]:
    """Klinger Volume Oscillator. Each row of data is [high, close, low, volume].

    Streaming dual EMA seeded from the first volume-flow value, plus a
    trend/cumulative-range accumulator. Output length is ``n - 1``.
    """
    n = len(data)
    if n <= 1:
        return []
    a_f = 2.0 / (fast + 1.0)
    a_s = 2.0 / (slow + 1.0)
    prev_hlc = float(data[0][0]) + float(data[0][1]) + float(data[0][2])
    trend = -1
    cm = 0.0
    ema_fast = 0.0; ema_slow = 0.0
    out: list[float] = [0.0] * (n - 1)
    for i in range(1, n):
        h = float(data[i][0]); c = float(data[i][1]); lo = float(data[i][2]); v = float(data[i][3])
        hlc = h + c + lo
        dm = h - lo
        if hlc > prev_hlc and trend != 1:
            trend = 1
            cm = float(data[i - 1][0]) - float(data[i - 1][2])
        elif hlc < prev_hlc and trend != 0:
            trend = 0
            cm = float(data[i - 1][0]) - float(data[i - 1][2])
        cm += dm
        sign = -1.0 if trend == 0 else 1.0
        vf = 0.0 if cm == 0 else v * abs(2.0 * dm / cm - 1.0) * 100.0 * sign
        if i == 1:
            ema_fast = vf
            ema_slow = vf
        else:
            ema_fast = (vf - ema_fast) * a_f + ema_fast
            ema_slow = (vf - ema_slow) * a_s + ema_slow
        out[i - 1] = ema_fast - ema_slow
        prev_hlc = hlc
    return out
