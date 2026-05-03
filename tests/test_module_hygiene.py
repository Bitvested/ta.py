"""Static checks on the ta_py package layout.

These tests parse module sources rather than importing them, so they
catch surface-level mistakes that import-time tests miss. The original
motivation is a Phase 1 bug where `fisher` and `avgwin` were defined
twice in the same file; the second definition silently shadowed the
first and dropped fixture coverage.
"""
import ast
import pathlib
from collections import Counter

PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / "ta_py"


def test_no_duplicate_top_level_defs() -> None:
    """No file in ta_py/ defines the same top-level function name twice."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text())
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        dups = [name for name, count in Counter(names).items() if count > 1]
        if dups:
            offenders[path.name] = dups
    assert not offenders, f"duplicate top-level defs: {offenders}"
