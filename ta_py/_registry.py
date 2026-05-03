"""Cross-indicator dispatch registry.

Mirrors ta.js's `src/_registry.js` pattern. The `__init__.py` populates
`registry` with every public function. Indicators that internally call
other indicators dereference them through this dict, which lets users
monkey-patch implementations at runtime:

    import ta_py
    ta_py._registry.registry["smma"] = my_custom_smma
    # ta_py.alligator now uses my_custom_smma

Mutation contract: keys may be reassigned (`registry["smma"] = my_fn`)
but must never be `del`'d. Indicator callsites index without `.get()`,
so a deleted key surfaces as `KeyError` at indicator runtime.
"""
from typing import Callable

registry: dict[str, Callable] = {}
