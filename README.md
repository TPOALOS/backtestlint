# backtestlint

[![tests](https://github.com/TPOALOS/backtestlint/actions/workflows/tests.yml/badge.svg)](https://github.com/TPOALOS/backtestlint/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

**A linter for look-ahead bias and data leakage in Python backtests.** It reads your code (it never runs it) and flags the
patterns that quietly let a backtest see the future: `shift(-1)`, centered rolling windows, shuffled train/test splits,
scalers fitted before the split, whole-sample z-scores, backward fills and more. A strategy that "works" because of one of these
lines is the most expensive kind of bug, so catch it before you trust the equity curve.

Pure Python standard library (the `ast` module). Works on `.py` files and Jupyter notebooks. Does not need pandas, numpy or scikit-learn installed.

```bash
python3 -m backtestlint path/to/strategy.py notebooks/ src/
```

```
examples/leaky_backtest.py:13:18: BL002 HIGH   [high confidence] rolling(center=True) averages rows before and after each point
examples/leaky_backtest.py:14:35: BL007 MEDIUM [medium confidence] 'prices.close' is combined with its own whole-sample mean()
examples/leaky_backtest.py:15:21: BL001 HIGH   [high confidence] shift(-1) pulls values from 1 row(s) in the future
examples/leaky_backtest.py:16:12: BL008 HIGH   [high confidence] method='bfill' fills gaps with future values
examples/leaky_backtest.py:20:9: BL006 MEDIUM [low confidence] transformer is fitted on 'the data' in a script that also splits the data
examples/leaky_backtest.py:21:10: BL001 HIGH   [high confidence] shift(-1) pulls values from 1 row(s) in the future
examples/leaky_backtest.py:22:40: BL005 HIGH   [high confidence] train_test_split shuffles by default
examples/leaky_backtest.py:26:13: BL004 MEDIUM [low confidence] 'signal' is multiplied by 'returns' but never shifted; this trades on same-bar information
backtestlint: 8 finding(s)
```

[`examples/leaky_backtest.py`](examples/leaky_backtest.py) is a deliberately flawed toy script (no real strategy or data).

## The 14 rules

| Rule | Severity | Flags |
|------|----------|-------|
| `BL001` negative-shift | HIGH | `shift(-n)`, `np.roll(x, -n)`: values from the future |
| `BL002` centered-window | HIGH | `rolling(..., center=True)` |
| `BL003` future-index-in-loop | MEDIUM | `data[i + 1]` / `iloc[i + 1]` inside a bar loop |
| `BL004` same-bar-execution | MEDIUM | a signal multiplied by returns without a shift |
| `BL005` shuffled-split | HIGH | `train_test_split` without `shuffle=False`, `KFold(shuffle=True)`, `ShuffleSplit` |
| `BL006` fit-before-split | MEDIUM | scaler / PCA / imputer fitted on all data in a script that later splits |
| `BL007` full-sample-statistic | MEDIUM | `(x - x.mean()) / x.std()` with whole-sample statistics |
| `BL008` backward-fill | HIGH | `bfill()`, `fillna(method="bfill")`, `interpolate()` |
| `BL009` future-join | HIGH | `merge_asof(direction="forward" / "nearest")` |
| `BL010` survivorship-dropna | MEDIUM | `dropna(axis=1)` deleting assets with gaps (often the delisted ones) |
| `BL011` future-return | MEDIUM | `pct_change(-n)`, `diff(-n)` |
| `BL012` last-row-in-loop | HIGH | `iloc[-1]` / `[-1]` inside a loop over the same data |
| `BL013` full-sample-rank | LOW | `qcut`, `rank(pct=True)` over the whole sample |
| `BL014` overlapping-cv | MEDIUM | plain `KFold` / `GroupKFold` on ordered data |

`python3 -m backtestlint rules` lists them and `python3 -m backtestlint explain BL001` explains why a rule exists and how to fix it.

## Usage

```bash
python3 -m backtestlint src/                              # files and folders (.py, .ipynb)
python3 -m backtestlint src/ --min-severity HIGH          # only the serious ones
python3 -m backtestlint src/ --min-confidence MEDIUM      # fewer guesses
python3 -m backtestlint src/ --select BL001 --select BL002
python3 -m backtestlint src/ --ignore BL013
python3 -m backtestlint src/ --format json                # machine-readable
python3 -m backtestlint src/ --format sarif > results.sarif   # for GitHub code scanning and other tools
python3 -m backtestlint src/ --exit-zero                  # report only, never fail
```

Exit codes: `0` clean, `1` findings, `2` a file could not be analysed or the options were wrong.

### Silencing a line you did it on purpose

Forward returns are fine as **labels** for a model (just never as features). Mark deliberate uses:

```python
y = close.pct_change().shift(-1)        # backtestlint: ignore[BL001]
y = close.pct_change().shift(-1)        # backtestlint: ignore          (all rules on this line)
```

and put `# backtestlint: ignore-file` on a line of its own to skip a whole file.

### Notebooks

`.ipynb` files are linted cell by cell (code cells only, IPython `%magics` and `!shell` lines are ignored) and the cell number is added to each message.

### In CI

```yaml
- run: python -m backtestlint src --min-severity MEDIUM
```

## Honest limitations

- **It is a heuristic linter, not a proof.** It looks at syntax and common names (`signal`, `position`, `ret`, `scaler` ...), without data-flow analysis or running your code.
  It will miss leaks it does not know and can flag harmless code; the confidence column tells you how sure each rule is. Rules named after variables (BL004, BL006, BL007) are the guessiest.
- A clean result does **not** mean your backtest is free of bias. Survivorship in your data source, restated fundamentals, execution assumptions, and costs are outside what a code linter can see.
- Patterns are pandas/numpy/scikit-learn flavoured, because that is what most backtests use; other libraries are only caught where they use the same method names.
- This is research tooling. Nothing here is investment advice or a statement about any strategy.

## How it works

`backtestlint/rules.py` holds the rule catalogue, `linter.py` walks the syntax tree and `cli.py` is the command line. Tests build small code snippets for every rule
(positive and negative), check suppression, notebooks, SARIF, and include a realistic walk-forward script that must stay clean.

```bash
python3 -m unittest -v        # 53 tests
```

## License

MIT (c) TPOALOS
