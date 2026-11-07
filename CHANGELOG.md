# Changelog

## 1.0.0

First release.

- 14 rules for look-ahead bias and data leakage (negative shifts, centered windows, same-bar execution, shuffled splits, fit-before-split, full-sample statistics, backward fill, forward joins, survivorship, and more).
- AST-based; Python files and Jupyter notebooks (cell numbers in messages).
- Text, JSON and SARIF output; inline `# backtestlint: ignore[RULE]` and `# backtestlint: ignore-file`.
- Severity and confidence filters; `rules` and `explain` commands.
