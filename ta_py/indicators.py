"""Indicator functions: trend, momentum, volume, volatility."""
import math
from typing import Callable

from ._registry import registry


def rsi(data: list[float], l: int = 14) -> list[float]:
    """Relative Strength Index (Wilder smoothing).

    Matches ta.js 2.0's ``rsi``: gains/losses are seeded as the mean of the
    first ``l`` raw gains/losses, then smoothed via Wilder's recurrence
    ``avg = (avg * (l - 1) + cur) / l``. Output length is ``n - l``.
    """
    n = len(data)
    if n <= l:
        return []
    gain = 0.0; loss = 0.0
    for i in range(1, l + 1):
        d = float(data[i]) - float(data[i - 1])
        if d > 0:
            gain += d
        else:
            loss -= d
    avg_g = gain / l
    avg_l = loss / l
    def _rsi_val(g: float, lo: float) -> float:
        if g == 0.0 and lo == 0.0:
            return float("nan")
        if lo == 0.0:
            return 100.0
        return 100.0 - 100.0 / (1 + g / lo)
    out: list[float] = [_rsi_val(avg_g, avg_l)]
    for i in range(l + 1, n):
        d = float(data[i]) - float(data[i - 1])
        avg_g = (avg_g * (l - 1) + (d if d > 0 else 0.0)) / l
        avg_l = (avg_l * (l - 1) + (-d if d < 0 else 0.0)) / l
        out.append(_rsi_val(avg_g, avg_l))
    return out
def wrsi(data: list[float], l: int = 14) -> list[float]:
    """Wilder's RSI. Alias for :func:`rsi` (matches ta.js 2.0)."""
    return rsi(data, l)
def tsi(data: list[float], long: int = 25, short: int = 13, sig: int = 13) -> list[list[float]]:
    """True Strength Index."""
    _ema = registry["ema"]  # alias once per call (registry contract: keys never deleted)
    mo = []; ab = []; ts = []; tsi = [];
    for i in range(1, len(data)):
        mo.append(data[i] - data[i - 1]);
        ab.append(abs(data[i] - data[i - 1]));
    sma1 = _ema(mo, long); sma2 = _ema(ab, long);
    ema1 = _ema(sma1, short); ema2 = _ema(sma2, short);
    for i in range(len(ema1)):
        ts.append(ema1[i] / ema2[i]);
    tma = _ema(ts, sig);
    ts = ts[(len(ts)-len(tma)):];
    for i in range(len(tma)):
        tsi.append([tma[i], ts[i]]);
    return tsi;
def macd(data: list[float], l1: int = 12, l2: int = 26) -> list[float]:
    """Moving Average Convergence Divergence (MACD line)."""
    _ema = registry["ema"]  # alias once per call (registry contract: keys never deleted)
    if l1 > l2: [l1, l2] = [l2, l1];
    em1 = _ema(data[:], l1); em2 = _ema(data[:], l2);
    em1 = em1[-(len(em2) - len(em1)):]; emf = [];
    for i in range(len(em1)):
        emf.append(em1[i] - em2[i]);
    return emf;
def macd_signal(data: list[float], l1: int = 12, l2: int = 26, l3: int = 9) -> list[float]:
    """MACD signal line (EMA of MACD)."""
    _ema = registry["ema"]  # alias once per call (registry contract: keys never deleted)
    return _ema(macd(data, l1, l2), l3);
def macd_bars(data: list[float], l1: int = 12, l2: int = 26, l3: int = 9) -> list[float]:
    """MACD histogram bars (MACD minus signal)."""
    _ema = registry["ema"]  # alias once per call (registry contract: keys never deleted)
    ma = macd(data, l1, l2); mas = _ema(ma, l3); ret = [];
    ma = ma[len(ma)-len(mas):];
    for i in range(len(ma)): ret.append(ma[i] - mas[i]);
    return ret;
def atr(data: list[list[float]], l1: int = 14) -> list[float]:
    """Average True Range. Each row of data is [high, close, low].

    Output length is ``n - l1 + 1`` (matching SMA shape).
    """
    n = len(data)
    if n < l1:
        return []
    prev = float(data[0][0]) - float(data[0][2])
    out: list[float] = []
    if l1 == 1:
        out.append(prev)
    for i in range(1, n):
        cprev = float(data[i - 1][1])
        h = float(data[i][0]); lo = float(data[i][2])
        tr = max(h - lo, abs(h - cprev), abs(lo - cprev))
        prev = (prev * (l1 - 1) + tr) / l1
        if i >= l1 - 1:
            out.append(prev)
    return out

def mfi(data: list[list[float]], l1: int = 14) -> list[float]:
    """Money Flow Index. Each row of data is [high, low, close, volume].

    Output length is ``n - l1``.
    """
    n = len(data)
    if n <= l1:
        return []
    pos = [0.0] * n
    neg = [0.0] * n
    prev_tp = (float(data[0][0]) + float(data[0][1]) + float(data[0][2])) / 3.0
    for i in range(1, n):
        h = float(data[i][0]); lo = float(data[i][1]); c = float(data[i][2]); v = float(data[i][3])
        tp = (h + lo + c) / 3.0
        flow = tp * v
        if tp > prev_tp:
            pos[i] = flow
        elif tp < prev_tp:
            neg[i] = flow
        prev_tp = tp
    sum_p = 0.0; sum_n = 0.0
    for i in range(1, l1 + 1):
        sum_p += pos[i]; sum_n += neg[i]
    def _mfi_val(p: float, ng: float) -> float:
        if p == 0.0 and ng == 0.0:
            return float("nan")
        if ng == 0.0:
            return 100.0
        return 100.0 * p / (p + ng)
    out: list[float] = [_mfi_val(sum_p, sum_n)]
    for i in range(l1 + 1, n):
        sum_p += pos[i] - pos[i - l1]
        sum_n += neg[i] - neg[i - l1]
        out.append(_mfi_val(sum_p, sum_n))
    return out
def roc(data: list[float], l1: int = 14) -> list[float]:
    """Rate of Change."""
    pl = []; roc = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            roc.append((pl[len(pl)-1] - pl[0]) / pl[0]);
            pl = pl[1:];
    return roc;
def mom(data: list[float], l1: int = 10, p: bool = False) -> list[float]:
    """Momentum (absolute or percentage)."""
    mom = [];
    for i in range(l1-1, len(data)):
        mom.append(data[i] / data[i - (l1-1)] * 100) if p == True else mom.append(data[i] - data[i - (l1 - 1)]);
    return mom
def kst(data: list[float], r1: int = 10, s1: int = 10, r2: int = 15, s2: int = 10, r3: int = 20, s3: int = 10, r4: int = 30, s4: int = 15, sig: int = 9) -> list[list[float]]:
    """Know Sure Thing oscillator."""
    _sma = registry["sma"]  # alias once per call (registry contract: keys never deleted)
    ks = []; fs = []; ms = (max(r1, r2, r3, r4) + max(s1, s2, s3, s4));
    for i in range(ms, len(data)+1):
        rcma1 = roc(data[i-ms:i], r1); rcma2 = roc(data[i-ms:i], r2); rcma3 = roc(data[i-ms:i], r3); rcma4 = roc(data[i-ms:i], r4);
        rcma1 = _sma(rcma1, s1); rcma2 = _sma(rcma2, s2); rcma3 = _sma(rcma3, s3); rcma4 = _sma(rcma4, s4);
        ks.append(rcma1[len(rcma1)-1] + rcma2[len(rcma2)-1] + rcma3[len(rcma3)-1] + rcma4[len(rcma4)-1]);
    sl = _sma(ks[:], sig);
    ks = ks[(len(ks) - len(sl)):];
    for i in range(len(ks)): fs.append([ks[i], sl[i]]);
    return fs;
def cop(data: list[float], l1: int = 11, l2: int = 14, l3: int = 10) -> list[float]:
    """Coppock Curve."""
    m = max(l1, l2); co = [];
    if l1 > l2: [l1, l2] = [l2, l1];
    for i in range(m + l3, len(data)):
        r1 = roc(data[i-(m+l3):i], l1); r2 = roc(data[i-(m+l3):i], l2); tmp = [];
        r1 = r1[(len(r1) - len(r2)):];
        for a in range(len(r1)):
            tmp.append(r1[a] + r2[a]);
        tmp = registry["wma"](tmp[:], l3);
        co.append(tmp[len(tmp)-1]);
    return co;
def obv(data: list[list[float]]) -> list[float]:
    """On-Balance Volume."""
    obv = [0];
    for i in range(1, len(data)):
        if(data[i][1] > data[i-1][1]): obv.append(obv[len(obv)-1] + data[i][0]);
        if(data[i][1] < data[i-1][1]): obv.append(obv[len(obv)-1] - data[i][0]);
        if(data[i][1] == data[i-1][1]): obv.append(obv[len(obv)-1]);
    return obv;
def vwap(data: list[list[float]], l1: int = 0) -> list[float]:
    """Volume-Weighted Average Price."""
    l1 = l1 if l1 > 0 else len(data); pl = []; vwap = [];
    for i in range(len(data)):
        pl.append([(data[i][0] * data[i][1]), data[i][1]]);
        if(len(pl) >= l1):
            totalv = 0; totalp = 0;
            for a in range(len(pl)):
                totalv += pl[a][1];
                totalp += pl[a][0];
            vwap.append(totalp/totalv);
            pl = pl[1:];
    return vwap;
def ichimoku(data: list[list[float]], l1: int = 9, l2: int = 26, l3: int = 52, l4: int = 26) -> list[list[float]]:
    """Ichimoku Cloud."""
    cloud = []; place = []; pl = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if(len(pl) >= l3):
            highs = []; lows = [];
            for a in range(len(pl)):
                highs.append(float(pl[a][0]));
                lows.append(float(pl[a][2]));
            tsen = (max(highs[len(highs)-l1:len(highs)]) + (min(lows[len(lows)-l1:len(lows)]))) / 2.0;
            ksen = (max(highs[len(highs)-l2:len(highs)]) + (min(lows[len(lows)-l2:len(lows)]))) / 2.0;
            senka = float(data[i][1]) + ksen;
            senkb = (max(highs[len(highs)-l3:len(highs)]) + (min(lows[len(lows)-l2:len(lows)]))) / 2.0;
            chik = float(data[i][1]);
            place.append([tsen, ksen, senka, senkb, chik]);
            pl = pl[1:];
    for i in range(l4, len(place)-l4):
        cloud.append([place[i][0], place[i][1], place[i+l4][2], place[i+l4][3], place[i-l4][4]]);
    return cloud;
def fractals(data: list[list[float]]) -> list[list[bool]]:
    """Bill Williams Fractals."""
    fractals = [[False, False], [False, False]];
    for i in range(2, len(data)-2):
        up = True if (data[i-2][0] < data[i][0] and data[i-1][0] < data[i][0] and data[i][0] > data[i+1][0] and data[i][0] > data[i+2][0]) else False
        down = True if (data[i-2][1] > data[i][1] and data[i-1][1] > data[i][1] and data[i][1] < data[i+1][1] and data[i][1] < data[i+2][1]) else False
        fractals.append([up, down]);
    fractals.append([False, False]);
    fractals.append([False, False]);
    return fractals;
def cross(d1: list[float], d2: list[float]) -> list[dict]:
    """Crossover indices between two series."""
    d1 = d1[len(d1)-len(d2):];
    cross = (d1[0] > d2[0]);
    indexes = [];
    for i in range(len(d1)):
        if d1[i] < d2[i] and cross:
            indexes.append({"index": i, "cross": False});
            cross = False;
        if d1[i] > d2[i] and cross == False:
            indexes.append({"index": i, "cross": True});
            cross = True;
    return indexes;
def halftrend(data: list[list[float]], atrlen: int, amplitude: int, deviation: int) -> list[list[float | str]]:
    """HalfTrend trend indicator."""
    out = []; nexttrend = [0]; trend = [0]; up = [0]; down = [0]; direction = None;
    for i in range(atrlen, len(data)):
        pl = data[i-atrlen:i];
        maxlow = pl[len(pl)-2][2];
        minhigh = pl[len(pl)-2][0];
        atr2 = atr(pl, atrlen);
        atr2 = atr2[len(atr2)-1] / 2;
        dev = deviation * atr2;
        highprice = max(pl[len(pl)-2][0], pl[len(pl)-1][0]);
        lowprice = min(pl[len(pl)-2][2], pl[len(pl)-1][2]);
        highs = list(map(lambda x: x[0], pl[len(pl)-amplitude:len(pl)]));
        lows = list(map(lambda x: x[2], pl[len(pl)-amplitude:len(pl)]));
        highma = registry["sma"](highs, amplitude);
        lowma = registry["sma"](lows, amplitude);
        if nexttrend[len(nexttrend)-1] == 1:
            maxlow = max(lowprice, maxlow);
            if highma[0] < maxlow and pl[len(pl)-1][1] < pl[len(pl)-2][2]:
                trend.append(1);
                nexttrend.append(0);
                minhigh = pl[len(pl)-2][0]
        else:
            minhigh = min(highprice, minhigh);
            if lowma[0] > minhigh and pl[len(pl)-1][1] < pl[len(pl)-2][0]:
                trend.append(0);
                nexttrend.append(1);
                maxlow = lowprice
        if trend[len(trend)-1] == 0:
            if not math.isnan(trend[len(trend)-2]) and trend[len(trend)-2] != 0:
                if math.isnan(down[len(down)-2]):
                    up.append(down[len(down-1)]);
                else:
                    up.append(down[len(down)-2]);
            else:
                if math.isnan(up[len(up)-2]):
                    up.append(maxlow);
                else:
                    up.append(max(up[len(up)-2], maxlow));
            direction = 'long';
            atrHigh = up[len(up)-1] + dev;
            atrLow = up[len(up)-1] - dev;
        else:
            if not math.isnan(trend[len(trend)-2] and trend[len(trend)-2] != 1):
                if math.isnan(up[len(up)-2]):
                    down.append(up[len(up)-1]);
                else:
                    down.append(up[len(up)-2]);
            else:
                if math.isnan(down[len(down)-2]):
                    down.append(minhigh);
                else:
                    down.append(min(minhigh, down[len(down)-2]));
            direction = 'short';
            atrHigh = down[len(down)-1] + dev;
            atrLow = down[len(down)-1] - dev;
        if trend[len(trend)-1] == 0:
            out.append([atrHigh, up[len(up)-1], atrLow, direction]);
        else:
            out.append([atrHigh, down[len(down)-1], atrLow, direction]);
    return out
def zigzag(data: list[float] | list[list[float]], perc: float = 0.05) -> list[float]:
    """ZigZag pivot detector."""
    indexes = []; min = float('inf'); max = float('-inf');
    lmin = False; lmax = False;
    if isinstance(data[0], list):
        for i in range(len(data)):
            if lmin:
                if min >= data[i][1]: min = data[i][1];
                if indexes[len(indexes)-1]['value'] >= data[i][1]:
                    indexes[len(indexes)-1]['value'] = data[i][1];
                    indexes[len(indexes)-1]['index'] = i;
                    continue
                try:
                    hdif = (data[i][0]-min)/min;
                except ZeroDivisionError:
                    hdif = 100
                if hdif > perc:
                    indexes.append({'index': i, 'value': data[i][0]});
                    lmax = True;
                    lmin = False;
                    min = float('inf');
            elif lmax:
                if max <= data[i][1]: max = data[i][1];
                if indexes[len(indexes)-1]['value'] <= data[i][0]:
                    indexes[len(indexes)-1]['value'] = data[i][0];
                    indexes[len(indexes)-1]['index'] = i;
                    continue
                try:
                    ldif = (max-data[i][1])/data[i][1];
                except ZeroDivisionError:
                    ldif = 100
                if ldif > perc:
                    indexes.append({'index': i, 'value': data[i][1]});
                    lmin = True;
                    lmax = False;
                    max = float('-inf');
            else:
                if min >= data[i][0]: min = data[i][1];
                if max <= data[i][0]: max = data[i][0];
                if i == 0: continue
                try:
                    hdif = (data[i][0]-min)/min;
                except ZeroDivisionError:
                    hdif = 100
                try:
                    ldif = (max-data[i][1])/max;
                except ZeroDivisionError:
                    ldif = 100
                if ldif > perc and hdif < perc:
                    lmin = True;
                    indexes.append({'index': 0, 'value': data[0][0]});
                    indexes.append({'index': i, 'value': data[i][1]});
                elif hdif > perc and ldif < perc:
                    lmax = True;
                    indexes.append({'index': 0, 'value': data[0][1]});
                    indexes.append({'index': i, 'value': data[i][0]});
                else:
                    if ldif > hdif:
                        lmin = True;
                        indexes.append({'index': 0, 'value': data[0][0]});
                        indexes.append({'index': i, 'value': data[i][1]});
                    else:
                        lmax = True;
                        indexes.append({'index': 0, 'value': data[0][1]});
                        indexes.append({'index': i, 'value': data[i][0]});
    else:
        for i in range(len(data)):
            if lmin:
                if min >= data[i]: min = data[i];
                if indexes[len(indexes)-1]['value'] >= data[i]:
                    indexes[len(indexes)-1]['value'] = data[i];
                    indexes[len(indexes)-1]['index'] = i;
                    continue
                try:
                    hdif = (data[i]-min)/min;
                except ZeroDivisionError:
                    hdif = 100
                if hdif > perc:
                    indexes.append({'index': i, 'value': data[i]});
                    lmax = True;
                    lmin = False;
                    min = float('inf');
            elif lmax:
                if max <= data[i]: max = data[i];
                if indexes[len(indexes)-1]['value'] <= data[i]:
                    indexes[len(indexes)-1]['value'] = data[i];
                    indexes[len(indexes)-1]['index'] = i;
                    continue
                ldif = (max-data[i])/data[i];
                if ldif > perc:
                    indexes.append({'index': i, 'value': data[i]});
                    lmin = True;
                    lmax = False;
                    max = float('-inf');
            else:
                if min >= data[i]: min = data[i];
                if max <= data[i]: max = data[i];
                if i == 0: continue
                try:
                    hdif = (data[i]-min)/min;
                except ZeroDivisionError:
                    hdif = 100
                try:
                    ldif = (max-data[i])/max;
                except ZeroDivisionError:
                    ldif = 100
                indexes.append({'index': 0, 'value': data[0]});
                indexes.append({'index': i, 'value': data[i]});
                if ldif > perc and hdif < perc:
                    lmin = True;
                elif hdif > perc and ldif < perc:
                    lmax = True;
                else:
                    if ldif > hdif:
                        lmin = True;
                    else:
                        lmax = True;
    final = [indexes[0]['value']];
    for i in range(1,len(indexes)):
        length = indexes[i]['index'] - indexes[i-1]['index'];
        delta = (indexes[i]['value'] - indexes[i-1]['value']) / length;
        for x in range(1, length+1):
            final.append(x*delta+indexes[i-1]['value']);
    return final;
def psar(data: list[list[float]], step: float = 0.02, maxi: float = 0.2) -> list[float]:
    """Parabolic SAR."""
    furthest = data[0]; up = True; accel = step; prev = data[0];
    sar = data[0][1]; extreme = data[0][0]; final = [sar];
    for i in range(1, len(data)):
        sar = sar + accel * (extreme - sar);
        if up:
            sar = min(sar, furthest[1], prev[1]);
            if data[i][0] > extreme:
                extreme = data[i][0];
                accel = min(accel+step, maxi);
        else:
            sar = max(sar, furthest[0], prev[0]);
            if data[i][1] < extreme:
                extreme = data[i][0];
                accel = min(accel + step, maxi);
        if (up and data[i][1] < sar) or (not up and data[i][0] > sar):
            accel = step;
            sar = extreme;
            up = not up;
            if not up:
                extreme = data[i][1];
            else:
                extreme = data[i][0];
        furthest = prev;
        prev = data[i];
        final.append(sar);
    return final;
def supertrend(data: list[list[float]], length: int = 20, multiplier: float = 3) -> list[list[float]]:
    """SuperTrend bands."""
    at = atr(data, length); trend = [];
    for i in range(len(at)):
        bar = i + length - 1
        mid = (data[bar][0] + data[bar][2]) / 2
        trend.append([mid + multiplier * at[i], mid - multiplier * at[i]]);
    return trend
def elderray(data: list[float], length: int = 13) -> list[list[float]]:
    """Elder-Ray Bull/Bear Power."""
    eld = [];
    for i in range(length, len(data)+1):
        pl = data[i-length:i];
        low = min(pl);
        high = max(pl);
        em = registry["ema"](pl, len(pl))
        eld.append([high-em[0],low-em[0]]);
    return eld;
def hv(data: list[float], length: int = 20) -> list[float]:
    """Historical Volatility."""
    hv = [];
    for i in range(length, len(data)+1):
        ss = registry["ssd"](data[i-length:i]);
        vari = ss[0] / length;
        hv.append(math.sqrt(vari));
    return hv;
def rvi(data: list[list[float]], l: int = 20) -> list[float]:
    """Relative Vigor Index."""
    num = []; dnom = []; rv = [];
    for i in range(3, len(data)):
        num.append((data[i][3]-data[i][0]+2*(data[i][3]-data[i-1][0])+2*(data[i][3]-data[i-2][0])+(data[i][3]-data[i-3][0]))/6);
        dnom.append((data[i][1]-data[i][2]+2*(data[i][1]-data[i-1][2])+2*(data[i][1]-data[i-2][2])+(data[i][1]-data[i-3][2]))/6);
        if len(num) >= l:
            sn = registry["sma"](num, l);
            dn = registry["sma"](dnom, l);
            rv.append(sn[0]/dn[0]);
            num = num[1:len(num)];
            dnom = dnom[1:len(dnom)];
    return rv;
def rvi_signal(rv: list[float]) -> list[float]:
    """Relative Vigor Index signal line."""
    sig = [];
    for i in range(3,len(rv)):
        sig.append((rv[i]+2*rv[i-1]+2*rv[i-2]+rv[i-3])/6);
    return sig;
def rsi_divergence(data: list[float], length: int, rs: Callable[[list[float], int], list[float]] | None = None) -> list[int]:
    """RSI divergence flags."""
    if rs is None:
        rs = registry["wrsi"]
    rd = rs(data, length); out = [];
    data = mom(data[length-1:], 2);
    for i in range(0, len(data)):
        if (data[i] > 0 and rd[i] < 0) or (data[i] < 0 and rd[i] > 0):
            out.append(1);
        else:
            out.append(0);
    return out;
def divergence(data1: list[float], data2: list[float]) -> list[int]:
    """Divergence flags between two series."""
    if len(data1) > len(data2):
        data1 = data1[len(data1)-len(data2):];
    if len(data2) > len(data1):
        data2 = data2[len(data2)-len(data1):];
    out = [];
    for i in range(1, len(data1)):
        if (data1[i] > data1[i-1] and data2[i] < data2[i-1]) or (data1[i] < data1[i-1] and data2[i] > data2[i-1]):
            out.append(1);
        else:
            out.append(0);
    return out;
def bop(data: list[list[float]], l1: int = 14) -> list[float]:
    """Balance of Power."""
    bo = [];
    for i in range(len(data)):
        bo.append((data[i][3] - data[i][0]) / (data[i][1] - data[i][2]) if (data[i][1] - data[i][2]) != 0 else 0);
    bo = registry["sma"](bo, l1);
    return bo;
def fi(data: list[list[float]], l1: int = 13) -> list[float]:
    """Force Index."""
    pl = []; ff = [];
    for i in range(1, len(data)):
        pl.append((data[i][0] - data[i-1][0]) * data[i][1]);
        if(len(pl) >= l1):
            vfi = registry["ema"](pl[:], l1);
            ff.append(vfi[len(vfi)-1]);
            pl = pl[1:];
    return ff;
def asi(data: list[list[float]]) -> list[float]:
    """Accumulation Swing Index."""
    pl = []; a = [];
    for i in range(1, len(data)):
        c = float(data[i][1]); y = float(data[i-1][1]); h = float(data[i][0]); hy = float(data[i-1][0]); cy = float(data[i-1][1]);
        l = float(data[i][2]); ly = float(data[i-1][2]); o = float(data[i][0]); oy = float(data[i-1][0]); t = max(float(data[i][0]), float(data[i-1][0])) - min(float(data[i][2]), float(data[i-1][2]));
        if(hy-c > ly-c): k = hy-c;
        else: k = ly-c;
        if((h - cy > l - cy) & (h - cy > h - l)): r = h - cy - (l - cy) / 2.0 + (cy - oy) / 4.0;
        if((l - cy > h - cy) & (l - cy > h - l)): r = l - cy - (h - cy) / 2.0 + (cy - oy) / 4.0;
        if((h - l > h - cy) & (h - l > l - cy)): r = h - l + (cy - oy) / 4.0;
        a.append(50.0 * ((cy - c + (cy - oy) / 2.0 + (c - o) / 2.0) / r) * k / t if r and t != 0 else 0);
    return a;
def alligator(data: list[float], jl: int = 13, tl: int = 8, ll: int = 5, js: int = 8, ts: int = 5, ls: int = 3) -> list[list[float]]:
    """Bill Williams Alligator (jaw, teeth, lips)."""
    ret = []; jaw = registry["smma"](data, jl); teeth = registry["smma"](data, tl); lips = registry["smma"](data, ll);
    teeth = teeth[len(teeth)-len(jaw):];
    lips = lips[len(lips)-len(jaw):];
    for i in range(len(jaw)-1, (js-2), -1):
        ret.append([jaw[i-(js-1)], teeth[i-(ts-1)], lips[i-(ls-1)]]);
    return ret;
def pr(data: list[float], l1: int = 14) -> list[float]:
    """Williams %R."""
    from collections import deque
    n = len(data)
    if n < l1:
        return []
    out: list[float] = []
    hi_dq: deque[int] = deque()
    lo_dq: deque[int] = deque()
    for i in range(n):
        v = float(data[i])
        while hi_dq and float(data[hi_dq[-1]]) <= v:
            hi_dq.pop()
        hi_dq.append(i)
        while lo_dq and float(data[lo_dq[-1]]) >= v:
            lo_dq.pop()
        lo_dq.append(i)
        if hi_dq[0] <= i - l1:
            hi_dq.popleft()
        if lo_dq[0] <= i - l1:
            lo_dq.popleft()
        if i >= l1 - 1:
            highd = float(data[hi_dq[0]])
            lowd = float(data[lo_dq[0]])
            rng = highd - lowd
            out.append(0.0 if rng == 0 else (highd - v) / rng * -100.0)
    return out
def stoch(data: list[list[float]], l1: int = 14, sd: int = 3, sk: int = 3) -> list[list[float]]:
    """Stochastic Oscillator (%K, %D)."""
    from collections import deque
    n = len(data)
    out: list[list[float]] = []
    # Effective window: original grows high/low over l1+sk+sd-1 bars before first output
    l1_eff = l1 + sk + sd - 1
    hi_dq: deque[int] = deque()
    lo_dq: deque[int] = deque()
    raw_buf: list[float] = [0.0] * sk
    sm_buf: list[float] = [0.0] * sd
    raw_sum = 0.0; sm_sum = 0.0
    raw_count = 0; sm_count = 0
    raw_idx = 0; sm_idx = 0
    for i in range(n):
        h = float(data[i][0]); lo = float(data[i][2])
        while hi_dq and float(data[hi_dq[-1]][0]) <= h:
            hi_dq.pop()
        hi_dq.append(i)
        while lo_dq and float(data[lo_dq[-1]][2]) >= lo:
            lo_dq.pop()
        lo_dq.append(i)
        if hi_dq[0] <= i - l1_eff:
            hi_dq.popleft()
        if lo_dq[0] <= i - l1_eff:
            lo_dq.popleft()
        if i < l1_eff - 1:
            continue
        highd = float(data[hi_dq[0]][0]); lowd = float(data[lo_dq[0]][2])
        rng = highd - lowd
        raw_k = 0.0 if rng == 0 else 100.0 * (float(data[i][1]) - lowd) / rng
        if raw_count < sk:
            raw_buf[raw_idx] = raw_k
            raw_sum += raw_k
            raw_count += 1
        else:
            raw_sum += raw_k - raw_buf[raw_idx]
            raw_buf[raw_idx] = raw_k
        raw_idx = (raw_idx + 1) % sk
        if raw_count < sk:
            continue
        sm_k = raw_sum / sk
        if sm_count < sd:
            sm_buf[sm_idx] = sm_k
            sm_sum += sm_k
            sm_count += 1
        else:
            sm_sum += sm_k - sm_buf[sm_idx]
            sm_buf[sm_idx] = sm_k
        sm_idx = (sm_idx + 1) % sd
        if sm_count < sd:
            continue
        out.append([sm_k, sm_sum / sd])
    return out
def fib(start: float, end: float) -> list[float]:
    """Fibonacci retracement levels."""
    return [start, (end-start)*.236+start, (end-start)*.382+start, (end-start)*.5+start, (end-start)*.618+start, (end-start)*.786+start, end, (end-start)*1.618+start, (end-start)*2.618+start, (end-start)*3.618+start, (end-start)*4.236+start]
def aroon_up(data: list[float], l1: int = 10) -> list[float]:
    """Aroon Up."""
    pl = []; aroon = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            hl = pl[:];
            aroon.append((100.0 * (l1-1-pl.index(max(hl))) / (l1-1)) if l1-1 != 0 else 0);
            pl = pl[1:];
    return aroon;
def aroon_down(data: list[float], l1: int = 10) -> list[float]:
    """Aroon Down."""
    pl = []; aroon = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            hl = pl[:];
            hl.reverse();
            aroon.append(100.0 * (l1-1-hl.index(min(hl))) / (l1-1) if l1-1 != 0 else 0);
            pl = pl[1:];
    return aroon;
def aroon_osc(data: list[float], l1: int = 10) -> list[float]:
    """Aroon Oscillator."""
    pl = []; aroon = [];
    u = aroon_up(data[:], l1); d = aroon_down(data[:], l1);
    for i in range(len(u)): aroon.append(u[i] - d[i]);
    return aroon;

def adl(data: list[list[float]]) -> list[float]:
    """Accumulation/Distribution Line. Each row of data is [high, close, low, volume]."""
    n = len(data)
    if n == 0:
        return []
    out: list[float] = [0.0] * n
    acc = 0.0
    for i in range(n):
        h = float(data[i][0]); c = float(data[i][1]); lo = float(data[i][2]); v = float(data[i][3])
        rng = h - lo
        mfm = 0.0 if rng == 0 else ((c - lo) - (h - c)) / rng
        acc += mfm * v
        out[i] = acc
    return out

def nvi(data: list[list[float]]) -> list[float]:
    """Negative Volume Index. Each row of data is [close, volume]. Seed = 1000."""
    n = len(data)
    if n == 0:
        return []
    out: list[float] = [0.0] * n
    out[0] = 1000.0
    for i in range(1, n):
        cprev = float(data[i - 1][0])
        if float(data[i][1]) < float(data[i - 1][1]) and cprev != 0:
            out[i] = out[i - 1] * float(data[i][0]) / cprev
        else:
            out[i] = out[i - 1]
    return out

def pvi(data: list[list[float]]) -> list[float]:
    """Positive Volume Index. Each row of data is [close, volume]. Seed = 1000."""
    n = len(data)
    if n == 0:
        return []
    out: list[float] = [0.0] * n
    out[0] = 1000.0
    for i in range(1, n):
        cprev = float(data[i - 1][0])
        if float(data[i][1]) > float(data[i - 1][1]) and cprev != 0:
            out[i] = out[i - 1] * float(data[i][0]) / cprev
        else:
            out[i] = out[i - 1]
    return out

def emv(data: list[list[float]]) -> list[float]:
    """Ease of Movement (scale=10000). Each row of data is [high, low, volume]."""
    n = len(data)
    if n < 2:
        return []
    out: list[float] = [0.0] * (n - 1)
    for i in range(1, n):
        h = float(data[i][0]); lo = float(data[i][1]); v = float(data[i][2])
        hp = float(data[i - 1][0]); lp = float(data[i - 1][1])
        move = (h + lo) / 2 - (hp + lp) / 2
        rng = h - lo
        out[i - 1] = float("nan") if (rng == 0 or v == 0) else move * rng * 10000 / v
    return out

def dpo(data: list[float], length: int = 21) -> list[float]:
    """Detrended Price Oscillator. ``DPO[k] = data[k + length-2 - length//2] - sma(data, length)[k]``."""
    _sma = registry["sma"]
    sm = _sma(data, length)
    if len(sm) == 0:
        return []
    off = length - 2 - length // 2
    return [float(data[i + off]) - sm[i] for i in range(len(sm))]

def pdm(data: list[list[float]], length: int = 14) -> list[float]:
    """Wilder +DM (average form). Each row of data is [high, close, low].

    Output length is ``n - length``.
    """
    n = len(data)
    if n < length + 1:
        return []
    s = 0.0
    for i in range(1, length + 1):
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        if up > dn and up > 0:
            s += up
    prev = s / length
    out: list[float] = [0.0] * (n - length)
    out[0] = prev
    for i in range(length + 1, n):
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        dm = up if (up > dn and up > 0) else 0.0
        prev = (prev * (length - 1) + dm) / length
        out[i - length] = prev
    return out

def mdm(data: list[list[float]], length: int = 14) -> list[float]:
    """Wilder -DM (average form). Each row of data is [high, close, low].

    Output length is ``n - length``.
    """
    n = len(data)
    if n < length + 1:
        return []
    s = 0.0
    for i in range(1, length + 1):
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        if dn > up and dn > 0:
            s += dn
    prev = s / length
    out: list[float] = [0.0] * (n - length)
    out[0] = prev
    for i in range(length + 1, n):
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        dm = dn if (dn > up and dn > 0) else 0.0
        prev = (prev * (length - 1) + dm) / length
        out[i - length] = prev
    return out

def mdi(data: list[list[float]], length: int = 14) -> list[float]:
    """-DI = 100 · smoothed(-DM) / smoothed(TR). Each row of data is [high, close, low].

    Mirror of pdi using -DM. Output length is ``n - length``.
    """
    n = len(data)
    if n < length + 1:
        return []
    sum_dm = 0.0; sum_tr = 0.0
    for i in range(1, length + 1):
        h = float(data[i][0]); lo = float(data[i][2]); cprev = float(data[i - 1][1])
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        if dn > up and dn > 0:
            sum_dm += dn
        sum_tr += max(h - lo, abs(h - cprev), abs(lo - cprev))
    dm = sum_dm / length
    tr = sum_tr / length
    out: list[float] = [0.0] * (n - length)
    out[0] = float("nan") if tr == 0 else 100.0 * dm / tr
    for i in range(length + 1, n):
        h = float(data[i][0]); lo = float(data[i][2]); cprev = float(data[i - 1][1])
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        cur_dm = dn if (dn > up and dn > 0) else 0.0
        cur_tr = max(h - lo, abs(h - cprev), abs(lo - cprev))
        dm = (dm * (length - 1) + cur_dm) / length
        tr = (tr * (length - 1) + cur_tr) / length
        out[i - length] = float("nan") if tr == 0 else 100.0 * dm / tr
    return out

def dx(data: list[list[float]], length: int = 14) -> list[float]:
    """Directional Movement Index. ``DX = 100 · |+DI − −DI| / (+DI + −DI)``.

    Each row of data is [high, close, low]. Output length is ``n - length``.
    """
    _pdi = registry["pdi"]; _mdi = registry["mdi"]
    p = _pdi(data, length)
    m = _mdi(data, length)
    out: list[float] = [0.0] * len(p)
    for i in range(len(p)):
        s = p[i] + m[i]
        out[i] = float("nan") if s == 0 else 100.0 * abs(p[i] - m[i]) / s
    return out

def apo(data: list[float], length1: int = 12, length2: int = 26) -> list[float]:
    """Absolute Price Oscillator. Thin wrapper for ``macd`` (TA-Lib's APO == MACD with EMA MA-type)."""
    return registry["macd"](data, length1, length2)

def cci(data: list[list[float]], length: int = 20) -> list[float]:
    """Lambert Commodity Channel Index. Each row of data is [high, close, low].

    ``TP = (H + C + L) / 3``; ``CCI = (TP − SMA(TP, length)) / (0.015 · MAD(TP, length))``.
    Output length is ``n - length + 1``.
    """
    n = len(data)
    if n < length:
        return []
    tp = [(float(data[i][0]) + float(data[i][1]) + float(data[i][2])) / 3.0 for i in range(n)]
    out: list[float] = [0.0] * (n - length + 1)
    for i in range(length, n + 1):
        s = 0.0
        for j in range(i - length, i):
            s += tp[j]
        sma_tp = s / length
        mad = 0.0
        for j in range(i - length, i):
            mad += abs(tp[j] - sma_tp)
        mad /= length
        out[i - length] = float("nan") if mad == 0 else (tp[i - 1] - sma_tp) / (0.015 * mad)
    return out

def mass(data: list[list[float]], length: int = 25) -> list[float]:
    """Mass Index. Rolling sum over ``length`` bars of EMA(range, 9) / EMA(EMA(range, 9), 9).

    Each row of data is [high, low].
    """
    _ema = registry["ema"]
    n = len(data)
    rng = [float(data[i][0]) - float(data[i][1]) for i in range(n)]
    e1 = _ema(rng, 9)
    if len(e1) < 9:
        return []
    e2 = _ema(e1, 9)
    if len(e2) < length:
        return []
    offset = len(e1) - len(e2)
    ratio = [float("nan") if e2[i] == 0 else e1[i + offset] / e2[i] for i in range(len(e2))]
    s = 0.0
    for i in range(length):
        s += ratio[i]
    out: list[float] = [0.0] * (len(ratio) - length + 1)
    out[0] = s
    for i in range(length, len(ratio)):
        s += ratio[i] - ratio[i - length]
        out[i - length + 1] = s
    return out

def ppo(data: list[float], length1: int = 12, length2: int = 26) -> list[float]:
    """Percentage Price Oscillator. ``ppo[i] = 100 · MACD[i] / EMA(data, max(length1, length2))[i]``."""
    if length1 > length2:
        length1, length2 = length2, length1
    _ema = registry["ema"]; _macd = registry["macd"]
    eb = _ema(data, length2)
    m = _macd(data, length1, length2)
    out: list[float] = [0.0] * len(m)
    for i in range(len(m)):
        out[i] = float("nan") if eb[i] == 0 else 100.0 * m[i] / eb[i]
    return out

def trix(data: list[float], length: int = 30) -> list[float]:
    """TRIX: rate of change of triple-EMA. ``trix[i] = 100·(EMA³[i] − EMA³[i−1])/EMA³[i−1]``."""
    _ema = registry["ema"]
    e1 = _ema(data, length)
    if len(e1) < length:
        return []
    e2 = _ema(e1, length)
    if len(e2) < length:
        return []
    e3 = _ema(e2, length)
    if len(e3) < 2:
        return []
    out: list[float] = [0.0] * (len(e3) - 1)
    for i in range(1, len(e3)):
        prev = e3[i - 1]
        out[i - 1] = float("nan") if prev == 0 else 100.0 * (e3[i] - prev) / prev
    return out

def natr(data: list[list[float]], length: int = 14) -> list[float]:
    """Normalized ATR. ``NATR[k] = 100 · ATR[k] / Close[k + length − 1]``.

    Each row of data is [high, close, low]. Output length matches atr's
    ``n - length + 1``.
    """
    _atr = registry["atr"]
    a = _atr(data, length)
    if len(a) == 0:
        return []
    out: list[float] = [0.0] * len(a)
    for i in range(len(a)):
        c = float(data[i + length - 1][1])
        out[i] = float("nan") if c == 0 else 100.0 * a[i] / c
    return out

def adxr(data: list[list[float]], length: int = 14) -> list[float]:
    """Average Directional Index Rating. ``ADXR[k] = (ADX[k] + ADX[k − length + 1]) / 2``.

    Each row of data is [high, close, low].
    """
    _adx = registry["adx"]
    a = _adx(data, length)
    if len(a) < length:
        return []
    out: list[float] = [0.0] * (len(a) - length + 1)
    for i in range(length - 1, len(a)):
        out[i - length + 1] = (a[i] + a[i - length + 1]) / 2.0
    return out

def adx(data: list[list[float]], length: int = 14) -> list[float]:
    """Average Directional Index. Wilder smoothing of DX over ``length`` bars.

    Each row of data is [high, close, low]. Output length is ``n - 2*length + 1``.
    """
    _dx = registry["dx"]
    d = _dx(data, length)
    if len(d) < length:
        return []
    s = 0.0
    for i in range(length):
        s += d[i]
    prev = s / length
    out: list[float] = [0.0] * (len(d) - length + 1)
    out[0] = prev
    for i in range(length, len(d)):
        prev = (prev * (length - 1) + d[i]) / length
        out[i - length + 1] = prev
    return out

def pdi(data: list[list[float]], length: int = 14) -> list[float]:
    """+DI = 100 · smoothed(+DM) / smoothed(TR). Each row of data is [high, close, low].

    Internal TR seed uses TA-Lib convention (mean of first ``length`` raw TRs),
    distinct from ta.atr's TR[0]-seeded Wilder convention. Output length is
    ``n - length``.
    """
    n = len(data)
    if n < length + 1:
        return []
    sum_dm = 0.0; sum_tr = 0.0
    for i in range(1, length + 1):
        h = float(data[i][0]); lo = float(data[i][2]); cprev = float(data[i - 1][1])
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        if up > dn and up > 0:
            sum_dm += up
        sum_tr += max(h - lo, abs(h - cprev), abs(lo - cprev))
    dm = sum_dm / length
    tr = sum_tr / length
    out: list[float] = [0.0] * (n - length)
    out[0] = float("nan") if tr == 0 else 100.0 * dm / tr
    for i in range(length + 1, n):
        h = float(data[i][0]); lo = float(data[i][2]); cprev = float(data[i - 1][1])
        up = float(data[i][0]) - float(data[i - 1][0])
        dn = float(data[i - 1][2]) - float(data[i][2])
        cur_dm = up if (up > dn and up > 0) else 0.0
        cur_tr = max(h - lo, abs(h - cprev), abs(lo - cprev))
        dm = (dm * (length - 1) + cur_dm) / length
        tr = (tr * (length - 1) + cur_tr) / length
        out[i - length] = float("nan") if tr == 0 else 100.0 * dm / tr
    return out

def ulcer(data: list[float], length: int = 14) -> list[float]:
    """Ulcer Index over ``length`` bars.

    For each bar, drawdown_pct = 100 * (data[i] - rolling_max) / rolling_max,
    where rolling_max spans the trailing ``length`` bars (inclusive).
    Ulcer[k] = sqrt(mean(drawdown_pct ** 2 over length bars)).
    """
    from collections import deque
    n = len(data)
    if n < length:
        return []
    hi_dq: deque[int] = deque()
    dd = [0.0] * n
    for i in range(n):
        v = float(data[i])
        while hi_dq and float(data[hi_dq[-1]]) <= v:
            hi_dq.pop()
        hi_dq.append(i)
        if hi_dq[0] <= i - length:
            hi_dq.popleft()
        hi = float(data[hi_dq[0]])
        dd[i] = 0.0 if hi == 0 else 100.0 * (v - hi) / hi
    out: list[float] = [0.0] * (n - length + 1)
    s = 0.0
    for i in range(length):
        s += dd[i] * dd[i]
    out[0] = (s / length) ** 0.5
    for i in range(length, n):
        s += dd[i] * dd[i] - dd[i - length] * dd[i - length]
        out[i - length + 1] = (s / length) ** 0.5
    return out
