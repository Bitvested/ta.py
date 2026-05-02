"""Oscillator indicators."""
import math
from ._registry import registry


def gator(
    data: list[float],
    jl: int = 13, tl: int = 8, ll: int = 5,
    js: int = 8, ts: int = 5, ls: int = 3,
) -> list[list[float]]:
    """Gator Oscillator (alligator-derived)."""
    ret = []; jaw = registry["smma"](data, jl); teeth = registry["smma"](data, tl); lips = registry["smma"](data, ll);
    teeth = teeth[len(teeth)-len(jaw):];
    lips = lips[len(lips)-len(jaw):];
    for i in range(len(jaw)-1, (js-2), -1):
        ret.append([jaw[i-(js-1)]-teeth[i-(ts-1)], -(abs(teeth[i-(ts-1)]-lips[i-(ls-1)]))]);
    return ret;


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
