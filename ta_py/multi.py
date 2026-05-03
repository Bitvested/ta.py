"""Multi-process Monte Carlo simulation.

Mirrors ta.js's ``multi/multi.js`` worker_threads-based parallel ``sim`` using
Python's stdlib :mod:`multiprocessing`. Each worker process runs one simulated
price path; results are aggregated in the parent.

The algorithm is identical to :func:`ta_py.sim` (geometric Brownian motion
projected forward over the rolling-window mean and stdev of returns). The
single-process ``sim`` and ``multi.sim`` produce statistically equivalent
output but not bit-identical sequences (independent RNG streams per worker).
"""
from __future__ import annotations

import math
import os
import random as _stdlib_random
from typing import Optional

from .statistics import sim as _sim_sequential


def _simulate_chunk(args: tuple[list[float], int, int]) -> list[list[float]]:
    """Run ``count`` Monte Carlo projections. Top-level (picklable) for multiprocessing.

    Each worker handles a chunk of simulations to amortize fork/spawn + IPC
    pickling overhead.
    """
    from .statistics import normsinv, sma, std

    data, length, count = args
    base = list(data)
    out: list[list[float]] = []
    for _ in range(count):
        projected = list(base)
        for _ in range(length):
            change = [
                (projected[y] - projected[y - 1]) / projected[y - 1]
                for y in range(1, len(projected))
            ]
            mean = sma(change, len(change))
            st = std(change)
            rando = normsinv(_stdlib_random.random())
            projected.append(
                projected[-1] * math.exp(mean[0] - (st * st) / 2 + st * rando)
            )
        out.append(projected)
    return out


def sim(
    data: list[float],
    l: int = 50,
    sims: int = 1000,
    perc: float = -1,
    processes: Optional[int] = None,
) -> list[list[float]] | list[float]:
    """Parallel Monte Carlo simulation.

    Drop-in for :func:`ta_py.sim` with the same arguments and return shape.
    ``processes`` defaults to :func:`os.cpu_count`. Falls back to the
    sequential implementation when multiprocessing is unavailable or when
    ``sims`` is small enough that fork/spawn overhead dominates.
    """
    if sims < 8:
        return _sim_sequential(data, l, sims, perc)

    try:
        from multiprocessing import Pool, get_context
    except ImportError:
        return _sim_sequential(data, l, sims, perc)

    n_proc = processes if processes is not None else (os.cpu_count() or 1)
    n_proc = max(1, min(n_proc, sims))

    # Distribute sims across workers in roughly equal chunks so each worker
    # makes one round-trip across the process boundary instead of `sims` round-trips.
    base, extra = divmod(sims, n_proc)
    chunks = [(data, l, base + (1 if i < extra else 0)) for i in range(n_proc)]

    try:
        ctx = get_context()
        with ctx.Pool(n_proc) as pool:
            chunked = pool.map(_simulate_chunk, chunks)
    except (OSError, ValueError):
        return _sim_sequential(data, l, sims, perc)

    sd = [path for chunk in chunked for path in chunk]

    if perc <= -1:
        return sd

    final = list(data)
    for i in range(len(data), len(sd[0])):
        col = sorted(row[i] for row in sd)
        final.append(col[round(len(col) * perc)])
    return final
