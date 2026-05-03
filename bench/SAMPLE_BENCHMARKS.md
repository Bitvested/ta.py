# Sample benchmark output

This file used to commit a snapshot of one specific benchmark run. It was
removed because:

1. **Hardware-specific.** Absolute numbers don't generalise across machines;
   only ratios within a single run on a single CPU mean anything.
2. **Configuration-specific.** Without all four optional adapters installed
   (`pip install -e .[all]`), most cells became `(skip)` and the file looked
   like a comparison table when it wasn't.
3. **Wrong framing.** ta.py is pure-Python by design; comparing absolute
   throughput against C-backed (TA-Lib) or NumPy-backed (pandas-ta, finta)
   libraries always loses, and that loss is exactly the trade-off the library
   asks you to make in exchange for zero install pain, zero native deps, and
   full Python-version portability. See the project README for the full pitch.

To produce numbers for **your** machine and **your** Python version:

```bash
cd bench
pip install -e .[all]
python run.py --sizes 1000,10000,100000 --md --out results/
```

Output goes to `results/BENCHMARKS.md`, which is git-ignored.
