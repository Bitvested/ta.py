"""Random / PRNG helpers."""
import math
import random as _stdlib_random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def prng(seed: str) -> Callable[[], float]:
    """Return a deterministic [0, 1) generator seeded by `seed` (str or int)."""
    h = (1779033703 ^ len(seed)) & 0xFFFFFFFF;
    for i in range(len(seed)):
        h = ((h ^ ord(seed[i])) * 3432918353) & 0xFFFFFFFF;
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF;
    h = ((h ^ (h >> 16)) * 2246822507) & 0xFFFFFFFF;
    h = ((h ^ (h >> 13)) * 3266489909) & 0xFFFFFFFF;
    state = [(h ^ (h >> 16)) & 0xFFFFFFFF];
    def gen():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF;
        t = state[0];
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF;
        t = (t ^ ((t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF)) & 0xFFFFFFFF;
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296;
    return gen;


def _draw(rng: Callable[[], float] | None) -> float:
    return rng() if rng is not None else _stdlib_random.random()


def pick(data: Sequence[T], rng: Callable[[], float] | None = None) -> T:
    """Pick one element from ``data`` using ``rng`` (defaults to stdlib random)."""
    return data[math.floor(_draw(rng) * math.floor(len(data)))]


def range_(min_v: float, max_v: float, rng: Callable[[], float] | None = None) -> int:
    """Random integer in ``[ceil(min_v), floor(max_v))``."""
    lo = math.ceil(min_v)
    hi = math.floor(max_v)
    return math.floor(_draw(rng) * (hi - lo) + lo)


def float_(min_v: float, max_v: float, rng: Callable[[], float] | None = None) -> float:
    """Random float in ``[min_v, max_v)``."""
    return _draw(rng) * (max_v - min_v) + min_v


def order(data: Sequence[T], rng: Callable[[], float] | None = None) -> list[T]:
    """Return ``data`` shuffled into a fresh list via repeated random picks."""
    pool = list(data)
    out: list[T] = []
    while pool:
        idx = math.floor(_draw(rng) * math.floor(len(pool)))
        out.append(pool[idx])
        pool.pop(idx)
    return out
