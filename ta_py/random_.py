"""Random / PRNG helpers."""
from typing import Callable


def prng(seed) -> Callable[[], float]:
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
