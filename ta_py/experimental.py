"""Experimental indicators. API may change in future releases."""

from .statistics import recent_low, recent_high


def support(d: list[float], hl: int = 0) -> dict:
    """Build a support line. Returns dict with 'calculate' callable, 'low', and 'index'."""
    hl = hl if hl else recent_low(d);
    findex = False; lowform = hl['value'];
    while findex == False:
        for i in range(hl['index'], len(d)):
            if hl['index']-i == 0: continue;
            newlow = (hl['value']-d[i])/(hl['index']-i);
            if newlow < lowform:
                lowform = newlow;
                index2 = i;
        if hl['index'] + 1 == index2 and index2 != len(d):
            hl['index'] = index2;
            lowform = min(d[:]);
            hl['value'] = d[hl['index']];
            findex = False;
        else:
            findex = True;
        if hl['index'] == len(d)-1: findex = True;
    if index2 == len(d)-1 or hl['index'] == len(d)-1:
        def calculate(pos):
            return pos*0+hl['value'];
        return {"calculate": calculate, "slope": 0, "lowest": hl['value'], "index": hl['index']};
    else:
        def calculate(pos):
            return pos*lowform+hl['value']
        return {"calculate": calculate, "slope": lowform, "lowest": hl['value'], "index": hl['index']};


def resistance(d: list[float], hl: int = 0) -> dict:
    """Build a resistance line. Returns dict with 'calculate' callable, 'high', and 'index'."""
    hl = hl if hl else recent_high(d);
    findex = False; highform = hl['value'];
    while findex == False:
        for i in range(hl['index'], len(d)):
            if hl['index']-i == 0: continue;
            newhigh = (d[i]-hl['value'])/(hl['index']-i);
            if newhigh < highform:
                highform = newhigh
                index2 = i;
        if hl['index']+1 == index2 and index2 != len(d)-1:
            hl['index'] = index2;
            highform = max(d[:]);
            hl['value'] = d[hl['index']];
            findex = False;
        else:
            findex = True;
        if hl['index'] == len(d)-1: findex = True;
    if index2 == len(d)-1 or hl['index'] == len(d)-1:
        highform = 0;
    if index2 == len(d)-1 or hl['index'] == len(d)-1:
        def calculate(pos):
            return pos*0+hl['value'];
        return {"calculate": calculate, "slope": 0, "highest": hl['value'], "index": hl['index']}
    else:
        def calculate(pos):
            return pos*-highform+hl['value'];
        return {"calculate": calculate, "slope": highform, "highest": hl['value'], "index": hl['index']};


def divergence_state(
    data1: list[float],
    data2: list[float],
    length: int,
    lb: int,
    smoother: int = 1,
    threshold_ex: float = 0.03,
    threshold_nm: float = 0.01,
) -> list[list[str]]:
    """Classify divergence state between two series across a lookback window."""
    if len(data1) > len(data2): data1 = data1[len(data1)-len(data2):];
    if len(data2) > len(data1): data2 = data2[len(data2)-len(data1):];
    out = [];
    for i in range(length, len(data1)):
        pl1 = data1[i-length:i+1];
        s1 = support(pl1, recent_low(pl1, lb)); s1d = s1['slope'] / s1['lowest'];
        r1 = resistance(pl1, recent_high(pl1, lb)); r1d = r1['slope'] / r1['highest'];
        pl2 = data2[i-length:i+1];
        s2 = support(pl2, recent_low(pl2, lb)); s2d = s2['slope'] / s2['lowest'];
        r2 = resistance(pl2, recent_high(pl2, lb)); r2d = r2['slope'] / r2['highest'];
        if (data1[i] > data1[i-smoother] and data2[i] < data2[i-smoother]) or (data1[i] < data1[i-smoother] and data2[i] > data2[i-smoother]):
            obj = [];
            if r1d < -threshold_ex and r2d > -threshold_nm: obj.append('exaggerated_bearish');
            if s1d < threshold_nm and s2d > threshold_ex: obj.append('exaggerated_bullish');
            if r1d < -threshold_nm and r2d < threshold_nm: obj.append('hidden_bearish');
            if s1d > threshold_nm and s2d < -threshold_nm: obj.append('hidden_bullish');
            if r1d > threshold_nm and r2d < -threshold_nm: obj.append('regular_bearish');
            if s1d < -threshold_nm and s2d > threshold_nm: obj.append('regular_bullish');
            if len(obj) <= 0: obj.append('divergence');
            out.append(obj);
        else:
            out.append(['convergence']);
    return out;
