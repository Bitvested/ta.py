"""Chart-type transformations: Heikin-Ashi and Renko."""


def ha(data: list[list[float]]) -> list[list[float]]:
    """Heikin-Ashi candles. Each row of data is [open, high, low, close].

    One row out per row in: bar 0 seeds the series instead of being transformed,
    so the loop starts at bar 1.
    """
    h = [[(float(data[0][0]) + float(data[0][3])) / 2, float(data[0][1]), float(data[0][2]), (float(data[0][0]) + float(data[0][1]) + float(data[0][2]) + float(data[0][3])) / 4.0]];
    for i in range(1, len(data)):
        h.append([(h[len(h)-1][0] + h[len(h)-1][3]) / 2, max(h[len(h)-1][0], h[len(h)-1][3], float(data[i][1])), min(h[len(h)-1][0], h[len(h)-1][3], float(data[i][2])), (float(data[i][0]) + float(data[i][1]) + float(data[i][2]) + float(data[i][3])) / 4.0]);
    return h;


def ren(data: list[list[float]], bs: float = 1) -> list[list[float]]:
    """Renko bricks. Each row of data is [high, low]."""
    re = []; decimals = len(str(bs-int(bs))[1:]);
    bh = round(round(float(data[0][0]) / float(bs) * (10.0 ** decimals)) / (10.0 ** decimals) * float(bs)); bl = bh - bs;
    for i in range(1, len(data)):
        if(data[i][0] > bh + bs):
            while (data[i][0] > bh + bs):
                re.append([bh,bh+bs,bh,bh+bs]);
                bh+=bs;
                bl+=bs;
        elif (data[i][1] < bl - bs):
            while (data[i][1] < bl - bs):
                re.append([bl,bl,bl-bs,bl-bs]);
                bh-=bs;
                bl-=bs;
    return re;
