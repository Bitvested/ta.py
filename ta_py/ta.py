"""Deprecated module. All public symbols now live in their category-specific modules.

Imports `from ta_py.ta import <name>` continue to work in 1.x for one release cycle
but emit no warning. Phase 5 (v2.0 release) will remove this shim.

`__all__` is rebound to the package's canonical list so `from ta_py.ta import *`
exports only public indicators — not the `math` module that the category modules
import internally.
"""
from .moving_averages import *  # noqa: F401,F403
from .statistics import *  # noqa: F401,F403
from .indicators import *  # noqa: F401,F403
from .oscillators import *  # noqa: F401,F403
from .bands import *  # noqa: F401,F403
from .chart_types import *  # noqa: F401,F403
from .misc import *  # noqa: F401,F403
from .experimental import *  # noqa: F401,F403
from .random_ import *  # noqa: F401,F403

from . import __all__ as __all__  # noqa: F401
