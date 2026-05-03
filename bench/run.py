"""Bench runner CLI."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from harness import bench, fmt_ops, fmt_time
from cases import CASES


def parse_args():
    p = argparse.ArgumentParser(description="ta.py cross-library bench harness")
    p.add_argument("--sizes", default="1000,10000",
                   help="Comma-separated input sizes (default: 1000,10000)")
    p.add_argument("--only", default=None,
                   help="Comma-separated case names to include (default: all)")
    p.add_argument("--reference", default="ta_py",
                   help="Adapter name to use as correctness reference (default: ta_py)")
    p.add_argument("--md", action="store_true", help="Write BENCHMARKS.md")
    p.add_argument("--json", action="store_true", help="Write results.json")
    p.add_argument("--out", default="results", help="Output directory (default: results/)")
    p.add_argument("--min-time-ms", type=int, default=200)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 1 size (200), only sma/std/bands, min-time 50ms")
    return p.parse_args()


def _filter_cases(only: str | None):
    if not only:
        return CASES
    keep = set(s.strip() for s in only.split(","))
    return [c for c in CASES if c.name in keep]


def _output_first(result):
    """Best-effort coercion of an adapter's return into a comparable Python list."""
    if result is None:
        return None
    if hasattr(result, "tolist"):
        try:
            return result.tolist()
        except Exception:
            pass
    if hasattr(result, "values"):  # pandas Series/DataFrame
        try:
            return result.values.tolist()
        except Exception:
            pass
    return list(result) if hasattr(result, "__iter__") else result


def _max_abs_delta(a, b) -> float | None:
    """Compare two outputs element-wise; return max abs delta, or None if shapes differ."""
    try:
        if hasattr(a, "__len__") and hasattr(b, "__len__"):
            n = min(len(a), len(b))
            mx = 0.0
            for i in range(n):
                ai, bi = a[i], b[i]
                if hasattr(ai, "__iter__") and hasattr(bi, "__iter__"):
                    sub = _max_abs_delta(list(ai), list(bi))
                    if sub is None: return None
                    mx = max(mx, sub)
                else:
                    try:
                        d = abs(float(ai) - float(bi))
                        if d == d:  # not NaN
                            mx = max(mx, d)
                    except (TypeError, ValueError):
                        return None
            return mx
    except Exception:
        return None
    return None


ADAPTER_ORDER = ["ta_py", "stdlib", "talib", "pandas_ta", "finta"]


def run():
    args = parse_args()
    if args.quick:
        args.sizes = "200"
        args.only = "sma,std,bands"
        args.min_time_ms = 50
    sizes = [int(s) for s in args.sizes.split(",")]
    cases = _filter_cases(args.only)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for case in cases:
        for size in sizes:
            data = case.generate(size)
            ref_fn = case.adapters.get(args.reference, lambda *_a: None)(data, case.length)
            ref_out = _output_first(ref_fn()) if ref_fn else None
            for lib in ADAPTER_ORDER:
                fn = case.adapters[lib](data, case.length)
                if fn is None:
                    print(f"  {case.name:10} size={size:>7} {lib:<10} (skip)")
                    results.append({"case": case.name, "size": size, "lib": lib, "status": "skip"})
                    continue
                try:
                    r = bench(fn, min_time_ms=args.min_time_ms)
                except Exception as exc:
                    print(f"  {case.name:10} size={size:>7} {lib:<10} ERROR: {exc}")
                    results.append({"case": case.name, "size": size, "lib": lib, "status": "error", "error": str(exc)})
                    continue
                delta = None
                if lib != args.reference and ref_out is not None:
                    delta = _max_abs_delta(ref_out, _output_first(fn()))
                line = f"  {case.name:10} size={size:>7} {lib:<10} {fmt_time(r.median_ns):>9}  {fmt_ops(r.ops_per_sec):>14}"
                if delta is not None:
                    line += f"  Δ={delta:.2e}" if isinstance(delta, float) else f"  Δ=N/A"
                print(line)
                results.append({
                    "case": case.name, "size": size, "lib": lib, "status": "ok",
                    "median_ns": r.median_ns, "p99_ns": r.p99_ns, "ops_per_sec": r.ops_per_sec,
                    "batch_size": r.batch_size, "samples": r.samples,
                    "delta_vs_reference": delta,
                })
    if args.json:
        path = out_dir / "results.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {path}")
    if args.md:
        _write_md(out_dir / "BENCHMARKS.md", results, args.reference)
        print(f"Wrote {out_dir / 'BENCHMARKS.md'}")


def _write_md(path: Path, results: list[dict], reference: str):
    by_case: dict[tuple[str, int], dict[str, dict]] = {}
    for r in results:
        by_case.setdefault((r["case"], r["size"]), {})[r["lib"]] = r
    lines = [
        "# Benchmarks", "",
        f"Reference library: `{reference}`. Δ column shows max absolute delta vs reference.",
        "Lower median is faster. `(skip)` = adapter not available or unsupported.", "",
    ]
    libs = ADAPTER_ORDER
    header = "| Case | Size | " + " | ".join(libs) + " |"
    sep = "|" + "------|" * (len(libs) + 2)
    lines.extend([header, sep])
    for (case, size), libmap in by_case.items():
        cells = []
        for lib in libs:
            r = libmap.get(lib, {"status": "skip"})
            if r["status"] == "skip":
                cells.append("(skip)")
            elif r["status"] == "error":
                cells.append(f"err: {r['error'][:30]}")
            else:
                d = r.get("delta_vs_reference")
                d_str = "" if d is None else f" Δ={d:.1e}" if isinstance(d, float) else ""
                cells.append(f"{fmt_time(r['median_ns'])}{d_str}")
        lines.append(f"| {case} | {size} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
