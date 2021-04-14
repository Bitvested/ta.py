def median(data, l=0):
    l = (l) if l > 0 else len(data); pl = []; med = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            tmp = pl.copy();
            tmp.sort(reverse=True);
            med.append(tmp[round((len(tmp)-1) / 2)]);
            pl = pl[1:];
    return med;
def rsi(data, l=0):
    l = (l) if l > 0 else len(data); pl = []; rs = [];
    for i in range(1, len(data)):
        pl.append(data[i] - data[i - 1]);
        if (len(pl)) >= l:
            gain = 0; loss = 0;
            for a in range(len(pl)):
                if pl[a] > 0: gain += pl[a];
                if pl[a] < 0: loss += abs(pl[a]);
            gain /= l; loss /= l; f = 100 - 100 / (1 + (gain / loss)) if loss != 0 else 100;
            rs.append(f);
            pl = pl[1:];
    return rs;
def sma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; sm = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            sm.append(sum(pl) / l);
            pl = pl[1:];
    return sm;
def smma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; sm = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            average = 0;
            for a in range(len(pl)): average += pl[a];
            f = average / l if len(sm) <= 0 else (average - sm[len(sm) - 1]) / l;
            sm.append(f);
            pl = pl[1:];
    sm = sm[1:];
    return sm;
def wma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; wm = []; weight = [];
    for i in range(1, l+1):
        weight.append(i);
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            average = 0;
            for q in range(1, len(pl)+1):
                average += pl[q-1] * q / sum(weight);
            wm.append(average);
            pl = pl[1:];
    return wm;
def ema(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; em = []; weight = 2 / (l + 1);
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            f = sum(pl) / l if len(em) <= 0 else (data[i] - em[len(em) - 1]) * weight + em[len(em) - 1]
            em.append(f);
            pl = pl[1:];
    return em;
def vwma(data, l=0):
    l = (l) if l > 0 else len(data) - 1; pl = []; wm = [];
    for i in range(len(data)):
        pl.append(data[i]);
        if (len(pl)) >= l:
            weight = 0; sum = 0;
            for a in range(len(pl)):
                sum += (pl[a][0] * pl[a][1]);
                weight += pl[a][1];
            wm.append(sum/weight);
            pl = pl[1:];
    return wm;


me = ema([1, 2, 3, 4, 5, 6, 10], 6);
print(me)
