"""Cross-indicator dispatch registry.

Mirrors ta.js's `src/_registry.js` pattern. The `__init__.py` populates
`registry` with every public function. Indicators that internally call
other indicators dereference them through this dict, which lets users
monkey-patch implementations at runtime:

    import ta_py
    ta_py._registry.registry["smma"] = my_custom_smma
    # ta_py.alligator now uses my_custom_smma

Until Task 16 wires call sites through this dict, it is populated but
unused.
"""
from typing import Callable, Dict

registry: Dict[str, Callable] = {}
