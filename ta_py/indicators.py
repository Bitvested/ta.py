"""Indicator functions: trend, momentum, volume, volatility."""
import math
from typing import Callable

from ._registry import registry


def rsi(data: list[float], l: int = 14) -> list[float]:
    """Relative Strength Index."""
    pl = []; rs = [];
    for i in range(1, len(data)):
        pl.append(data[i] - data[i - 1]);
        if (len(pl)) >= l-1:
            gain = 0.0; loss = 0.0;
            for a in range(len(pl)):
                if pl[a] > 0: gain += float(pl[a]);
                if pl[a] < 0: loss += abs(float(pl[a]));
            gain /= l; loss /= l;
            try:
                f = 100.0 - 100.0 / (1.0 + (gain/loss));
            except:
                f = 100.0
            rs.append(f);
            pl = pl[1:];
    return rs;
def wrsi(data: list[float], l: int = 14) -> list[float]:
    arrsi = []; u = []; d = [];
    for i in range(1,len(data)):
        if(data[i]-data[i-1]<0):
            d.append(abs(data[i]-data[i-1]));
            u.append(0.0);
        else:
            d.append(0.0);
            u.append(data[i]-data[i-1]);
    d = registry["wsma"](d, l); u = registry["wsma"](u, l);
    for i in range(0, len(d)):
        try:
            f = 100.0-100.0/(1.0+(u[i]/d[i]));
        except:
            f = 100.0;
        arrsi.append(f);
    return arrsi;
def tsi(data: list[float], long: int = 25, short: int = 13, sig: int = 13) -> list[list[float]]:
    mo = []; ab = []; ts = []; tsi = [];
    for i in range(1, len(data)):
        mo.append(data[i] - data[i - 1]);
        ab.append(abs(data[i] - data[i - 1]));
    sma1 = registry["ema"](mo, long); sma2 = registry["ema"](ab, long);
    ema1 = registry["ema"](sma1, short); ema2 = registry["ema"](sma2, short);
    for i in range(len(ema1)):
        ts.append(ema1[i] / ema2[i]);
    tma = registry["ema"](ts, sig);
    ts = ts[(len(ts)-len(tma)):];
    for i in range(len(tma)):
        tsi.append([tma[i], ts[i]]);
    return tsi;
def macd(data: list[float], l1: int = 12, l2: int = 26) -> list[float]:
    if l1 > l2: [l1, l2] = [l2, l1];
    em1 = registry["ema"](data[:], l1); em2 = registry["ema"](data[:], l2);
    em1 = em1[-(len(em2) - len(em1)):]; emf = [];
    for i in range(len(em1)):
        emf.append(em1[i] - em2[i]);
    return emf;
def macd_signal(data: list[float], l1: int = 12, l2: int = 26, l3: int = 9) -> list[float]:
    return registry["ema"](macd(data, l1, l2), l3);
def macd_bars(data: list[float], l1: int = 12, l2: int = 26, l3: int = 9) -> list[float]:
    ma = macd(data, l1, l2); mas = registry["ema"](ma, l3); ret = [];
    ma = ma[len(ma)-len(mas):];
    for i in range(len(ma)): ret.append(ma[i] - mas[i]);
    return ret;
def atr(data: list[list[float]], l1: int = 14) -> list[float]:
    atr = [float(data[0][0]) - float(data[0][2])];
    for i in range(1, len(data)):
        t0 = max((float(data[i][0]) - float(data[i - 1][1])), (float(data[i][2]) - float(data[i - 1][1])), (float(data[i][0]) - float(data[i][2])));
        atr.append((atr[len(atr)-1] * (l1 - 1) + t0) / l1)
    return atr;
def mfi(data: list[list[float]], l1: int = 14) -> list[float]:
    mfi = []; n = []; p = [];
    for i in range(len(data)):
        p.append(float(data[i][0])); n.append(float(data[i][1]));
        if(len(p) >= l1):
            mfi.append((100.0 - 100.0 / (1 + registry["sum"](p) / registry["sum"](n))));
            p = p[1:]; n = n[1:];
    return mfi;
def roc(data: list[float], l1: int = 14) -> list[float]:
    pl = []; roc = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            roc.append((pl[len(pl)-1] - pl[0]) / pl[0]);
            pl = pl[1:];
    return roc;
def mom(data: list[float], l1: int = 10, p: bool = False) -> list[float]:
    mom = [];
    for i in range(l1-1, len(data)):
        mom.append(data[i] / data[i - (l1-1)] * 100) if p == True else mom.append(data[i] - data[i - (l1 - 1)]);
    return mom
def kst(data: list[float], r1: int = 10, s1: int = 10, r2: int = 15, s2: int = 10, r3: int = 20, s3: int = 10, r4: int = 30, s4: int = 15, sig: int = 9) -> list[list[float]]:
    ks = []; fs = []; ms = (max(r1, r2, r3, r4) + max(s1, s2, s3, s4));
    for i in range(ms, len(data)+1):
        rcma1 = roc(data[i-ms:i], r1); rcma2 = roc(data[i-ms:i], r2); rcma3 = roc(data[i-ms:i], r3); rcma4 = roc(data[i-ms:i], r4);
        rcma1 = registry["sma"](rcma1, s1); rcma2 = registry["sma"](rcma2, s2); rcma3 = registry["sma"](rcma3, s3); rcma4 = registry["sma"](rcma4, s4);
        ks.append(rcma1[len(rcma1)-1] + rcma2[len(rcma2)-1] + rcma3[len(rcma3)-1] + rcma4[len(rcma4)-1]);
    sl = registry["sma"](ks[:], sig);
    ks = ks[(len(ks) - len(sl)):];
    for i in range(len(ks)): fs.append([ks[i], sl[i]]);
    return fs;
def cop(data: list[float], l1: int = 11, l2: int = 14, l3: int = 10) -> list[float]:
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
    obv = [0];
    for i in range(1, len(data)):
        if(data[i][1] > data[i-1][1]): obv.append(obv[len(obv)-1] + data[i][0]);
        if(data[i][1] < data[i-1][1]): obv.append(obv[len(obv)-1] - data[i][0]);
        if(data[i][1] == data[i-1][1]): obv.append(obv[len(obv)-1]);
    return obv;
def vwap(data: list[list[float]], l1: int = 0) -> list[float]:
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
    fractals = [[False, False], [False, False]];
    for i in range(2, len(data)-2):
        up = True if (data[i-2][0] < data[i][0] and data[i-1][0] < data[i][0] and data[i][0] > data[i+1][0] and data[i][0] > data[i+2][0]) else False
        down = True if (data[i-2][1] > data[i][1] and data[i-1][1] > data[i][1] and data[i][1] < data[i+1][1] and data[i][1] < data[i+2][1]) else False
        fractals.append([up, down]);
    fractals.append([False, False]);
    fractals.append([False, False]);
    return fractals;
def cross(d1: list[float], d2: list[float]) -> list[dict]:
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
    out = []; nexttrend = [0]; trend = [0]; up = [0]; down = [0]; direction = None;
    for i in range(atrlen, len(data)+1):
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
                except:
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
                except:
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
                except:
                    hdif = 100
                try:
                    ldif = (max-data[i][1])/max;
                except:
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
                except:
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
                except:
                    hdif = 100
                try:
                    ldif = (max-data[i])/max;
                except:
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
    at = atr(data, length); trend = [];
    for i in range(length-1, len(data)):
        trend.append([(data[i][0] + data[i][2]) / 2 + multiplier * at[i], (data[i][0] + data[i][2]) / 2 - multiplier * at[i]]);
    return trend
def elderray(data: list[float], length: int = 13) -> list[list[float]]:
    eld = [];
    for i in range(length, len(data)+1):
        pl = data[i-length:i];
        low = min(pl);
        high = max(pl);
        em = registry["ema"](pl, len(pl))
        eld.append([high-em[0],low-em[0]]);
    return eld;
def hv(data: list[float], length: int = 20) -> list[float]:
    hv = [];
    for i in range(length, len(data)+1):
        ss = registry["ssd"](data[i-length:i]);
        vari = ss[0] / length;
        hv.append(math.sqrt(vari));
    return hv;
def rvi(data: list[list[float]], l: int = 20) -> list[float]:
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
    sig = [];
    for i in range(3,len(rv)):
        sig.append((rv[i]+2*rv[i-1]+2*rv[i-2]+rv[i-3])/6);
    return sig;
def rsi_divergence(data: list[float], length: int, rs: Callable[[list[float], int], list[float]] | None = None) -> list[int]:
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
    bo = [];
    for i in range(len(data)):
        bo.append((data[i][3] - data[i][0]) / (data[i][1] - data[i][2]) if (data[i][1] - data[i][2]) != 0 else 0);
    bo = registry["sma"](bo, l1);
    return bo;
def fi(data: list[list[float]], l1: int = 13) -> list[float]:
    pl = []; ff = [];
    for i in range(1, len(data)):
        pl.append((data[i][0] - data[i-1][0]) * data[i][1]);
        if(len(pl) >= l1):
            vfi = registry["ema"](pl[:], l1);
            ff.append(vfi[len(vfi)-1]);
            pl = pl[1:];
    return ff;
def asi(data: list[list[float]]) -> list[float]:
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
    ret = []; jaw = registry["smma"](data, jl); teeth = registry["smma"](data, tl); lips = registry["smma"](data, ll);
    teeth = teeth[len(teeth)-len(jaw):];
    lips = lips[len(lips)-len(jaw):];
    for i in range(len(jaw)-1, (js-2), -1):
        ret.append([jaw[i-(js-1)], teeth[i-(ts-1)], lips[i-(ls-1)]]);
    return ret;
def pr(data: list[float], l1: int = 14) -> list[float]:
    n = []; pl = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            highd = max(pl[:]); lowd = min(pl[:]);
            n.append((highd - data[i]) / (highd - lowd) * -100.0 if highd-lowd != 0 else 0);
            pl = pl[1:];
    return n;
def stoch(data: list[list[float]], l1: int = 14, sd: int = 3, sk: int = 3) -> list[list[float]]:
    stoch = []; high = []; low = []; ka = [];
    for i in range(len(data)):
        high.append(float(data[i][0]));
        low.append(float(data[i][2]));
        if(len(high) >= l1):
            highd = max(high); lowd = min(low);
            k = 100.0 * (data[i][1] - lowd) / (highd - lowd) if highd - lowd != 0 else 0;
            ka.append(k);
        if(sk > 0 & len(ka) > sk):
            smoothedk = registry["sma"](ka[:], sk);
            ka.push(smoothedk[len(smoothedk)-1]);
        if(len(ka) - sk >= sd):
            d = registry["sma"](ka[:], sd);
            stoch.append([k, d[len(d)-1]]);
            high = high[1:];
            low = low[1:];
            ka = ka[1:];
    return stoch;
def fib(start: float, end: float) -> list[float]:
    return [start, (end-start)*.236+start, (end-start)*.382+start, (end-start)*.5+start, (end-start)*.618+start, (end-start)*.786+start, end, (end-start)*1.618+start, (end-start)*2.618+start, (end-start)*3.618+start, (end-start)*4.236+start]
def aroon_up(data: list[float], l1: int = 10) -> list[float]:
    pl = []; aroon = [];
    for i in range(len(data)):
        pl.append(float(data[i]));
        if(len(pl) >= l1):
            hl = pl[:];
            aroon.append((100.0 * (l1-1-pl.index(max(hl))) / (l1-1)) if l1-1 != 0 else 0);
            pl = pl[1:];
    return aroon;
def aroon_down(data: list[float], l1: int = 10) -> list[float]:
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
    pl = []; aroon = [];
    u = aroon_up(data[:], l1); d = aroon_down(data[:], l1);
    for i in range(len(u)): aroon.append(u[i] - d[i]);
    return aroon;
