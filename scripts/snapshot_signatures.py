"""Snapshot inspect.signature for every public name in ta_py.__all__ to JSON.

Used as a baseline diff before/after Phase 3 breaking-signature changes.
Run from repo root: python scripts/snapshot_signatures.py > tests/signatures_v1.json
"""
import inspect
import json
import sys

import ta_py


def describe(obj: object) -> dict:
    if not callable(obj):
        return {"kind": "value", "type": type(obj).__name__}
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return {"kind": "callable", "signature": None}
    params = []
    for p in sig.parameters.values():
        params.append({
            "name": p.name,
            "kind": p.kind.name,
            "default": "<empty>" if p.default is inspect.Parameter.empty else repr(p.default),
            "annotation": "<empty>" if p.annotation is inspect.Parameter.empty else str(p.annotation),
        })
    ret = "<empty>" if sig.return_annotation is inspect.Signature.empty else str(sig.return_annotation)
    return {"kind": "callable", "parameters": params, "return": ret}


def main() -> None:
    snapshot = {name: describe(getattr(ta_py, name)) for name in sorted(ta_py.__all__)}
    json.dump(snapshot, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
