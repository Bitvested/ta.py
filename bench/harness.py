"""Microbenchmark harness — pure stdlib, mirrors ta.js bench/harness.js shape."""
from __future__ import annotations
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Any


@dataclass
class BenchResult:
    median_ns: float
    p99_ns: float
    ops_per_sec: float
    batch_size: int
    samples: int
    raw_min_ns: float

    def to_dict(self) -> dict:
        return asdict(self)


def _measure_batch(fn: Callable[[], Any], batch: int) -> int:
    t0 = time.perf_counter_ns()
    for _ in range(batch):
        fn()
    return time.perf_counter_ns() - t0


def _calibrate_batch(fn: Callable[[], Any], target_ns: int = 1_000_000) -> int:
    """Find batch size so one batch takes ~target_ns (default 1ms)."""
    batch = 1
    while True:
        elapsed = _measure_batch(fn, batch)
        if elapsed >= target_ns or batch >= 1_000_000:
            return max(1, batch)
        # scale up; bias by 2x to avoid oscillation near boundary
        scale = max(2, int(target_ns / max(elapsed, 1)))
        batch = min(batch * scale, 1_000_000)


def bench(
    fn: Callable[[], Any],
    *,
    min_samples: int = 30,
    min_time_ms: int = 200,
    warmup_iters: int = 5,
) -> BenchResult:
    """Benchmark `fn` (zero-arg callable). Returns timing percentiles."""
    for _ in range(warmup_iters):
        fn()
    batch = _calibrate_batch(fn)
    samples_ns: list[float] = []
    deadline_ns = time.perf_counter_ns() + min_time_ms * 1_000_000
    while len(samples_ns) < min_samples or time.perf_counter_ns() < deadline_ns:
        elapsed = _measure_batch(fn, batch)
        samples_ns.append(elapsed / batch)
        if len(samples_ns) > 5_000:
            break
    samples_ns.sort()
    median = statistics.median(samples_ns)
    p99 = samples_ns[min(len(samples_ns) - 1, int(len(samples_ns) * 0.99))]
    return BenchResult(
        median_ns=median,
        p99_ns=p99,
        ops_per_sec=1e9 / median if median > 0 else float("inf"),
        batch_size=batch,
        samples=len(samples_ns),
        raw_min_ns=samples_ns[0],
    )


def fmt_ops(ops: float) -> str:
    if ops >= 1e9: return f"{ops/1e9:.2f}G ops/s"
    if ops >= 1e6: return f"{ops/1e6:.2f}M ops/s"
    if ops >= 1e3: return f"{ops/1e3:.2f}k ops/s"
    return f"{ops:.0f} ops/s"


def fmt_time(ns: float) -> str:
    if ns >= 1e9: return f"{ns/1e9:.2f}s"
    if ns >= 1e6: return f"{ns/1e6:.2f}ms"
    if ns >= 1e3: return f"{ns/1e3:.2f}µs"
    return f"{ns:.0f}ns"
