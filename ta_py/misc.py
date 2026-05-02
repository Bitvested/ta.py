"""Miscellaneous helpers."""


def times_up(data: list[float], l: int) -> list[int]:
    """Return [1, 0, …] with 1 where the previous `l` consecutive diffs in `data` are all positive."""
    out = [];
    for i in range(l, len(data)):
        up = 1;
        for x in range(i-l+1, i+1):
            if data[x-1] > data[x]:
                up = 0;
                break;
        out.append(up);
    return out;


def times_down(data: list[float], l: int) -> list[int]:
    """Return [1, 0, …] with 1 where the previous `l` consecutive diffs in `data` are all negative."""
    out = [];
    for i in range(l, len(data)):
        dn = 1;
        for x in range(i-l+1, i+1):
            if data[x-1] < data[x]:
                dn = 0;
                break;
        out.append(dn);
    return out;
