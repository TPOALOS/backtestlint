import textwrap
import unittest

from backtestlint.linter import lint_source


def ids(code):
    return [f.rule_id for f in lint_source(textwrap.dedent(code))]


class RuleCase(unittest.TestCase):
    def assertFlags(self, code, rule, count=1):
        found = ids(code)
        self.assertEqual(found.count(rule), count, "%s in %r -> %s" % (rule, code, found))

    def assertClean(self, code, rule=None):
        found = ids(code)
        self.assertEqual([r for r in found if rule is None or r == rule], [], "unexpected %s in %r" % (found, code))


class ShiftAndWindowTests(RuleCase):
    def test_negative_shift(self):
        for code in ("y = df.close.shift(-1)", "y = df['c'].shift(periods=-2)", "y = s.shift(-5)", "y = np.roll(x, -1)", "y = numpy.roll(x, -3)"):
            self.assertFlags(code, "BL001")

    def test_safe_shifts(self):
        for code in ("y = df.close.shift(1)", "y = df.close.shift()", "y = s.shift(periods=3)", "y = np.roll(x, 1)", "y = s.shift(-n)", "y = s.shift(k)"):
            self.assertClean(code, "BL001")

    def test_centered_window(self):
        self.assertFlags("m = s.rolling(5, center=True).mean()", "BL002")
        self.assertFlags("m = s.rolling(window=9, center=True).median()", "BL002")
        self.assertClean("m = s.rolling(5).mean()", "BL002")
        self.assertClean("m = s.rolling(5, center=False).mean()", "BL002")

    def test_forward_changes(self):
        self.assertFlags("r = s.pct_change(-1)", "BL011")
        self.assertFlags("r = s.diff(periods=-3)", "BL011")
        self.assertClean("r = s.pct_change()", "BL011")
        self.assertClean("r = s.pct_change(5)", "BL011")
        self.assertClean("r = s.diff(2)", "BL011")


class LoopIndexTests(RuleCase):
    def test_future_index(self):
        self.assertFlags("for i in range(len(df)):\n    x = df.iloc[i + 1]", "BL003")
        self.assertFlags("for i in range(n):\n    x = close[i + 2]", "BL003")
        self.assertFlags("for t in range(len(df)):\n    nxt = df.close[t+1]", "BL003")

    def test_safe_index(self):
        self.assertClean("for i in range(len(df)):\n    x = df.iloc[i]\n    y = df.iloc[i - 1]\n    z = df.iloc[i + 0]", "BL003")
        self.assertClean("i = 3\nx = df.iloc[i + 1]", "BL003")
        self.assertClean("for j in range(n):\n    x = df.iloc[i + 1]", "BL003")

    def test_last_row_in_loop(self):
        self.assertFlags("for i in range(len(df)):\n    cur = df.iloc[-1]", "BL012")
        self.assertFlags("for i in range(len(df)):\n    cur = df.close[-1]", "BL012")
        self.assertFlags("for idx, row in df.iterrows():\n    last = df.iloc[-1]", "BL012")
        self.assertClean("last = df.iloc[-1]", "BL012")
        self.assertClean("for i in range(len(df)):\n    cur = other.iloc[-1]", "BL012")
        self.assertClean("for i in range(len(df)):\n    cur = df.iloc[i]", "BL012")


class SignalTests(RuleCase):
    def test_same_bar_execution(self):
        self.assertFlags("sig = (df.c > df.ma).astype(int)\nstrat = sig * df.ret", "BL004")
        self.assertFlags("pnl = weights * returns", "BL004")
        self.assertFlags("pnl = ret * position", "BL004")
        self.assertFlags("x = signal * df.close.pct_change()", "BL004")

    def test_shifted_signal_is_fine(self):
        self.assertClean("sig = raw.shift(1)\nstrat = sig * ret", "BL004")
        self.assertClean("pos = sig.shift(1)\nr = pos * returns", "BL004")
        self.assertClean("r = signal.shift(1) * returns", "BL004")
        self.assertClean("w = w0.shift(periods=2)\nr = w * returns", "BL004")
        self.assertClean("r = position_size * volatility", "BL004")

    def test_negative_shift_does_not_count_as_a_proper_shift(self):
        self.assertFlags("sig = raw.shift(-1)\nstrat = sig * ret", "BL004")


class SplitTests(RuleCase):
    def test_shuffled_splits(self):
        self.assertFlags("a = train_test_split(X, y)", "BL005")
        self.assertFlags("a = train_test_split(X, y, test_size=0.2, shuffle=True)", "BL005")
        self.assertFlags("cv = KFold(n_splits=5, shuffle=True)", "BL005")
        self.assertFlags("cv = ShuffleSplit(n_splits=5)", "BL005")
        self.assertFlags("cv = StratifiedShuffleSplit(n_splits=3)", "BL005")

    def test_ordered_splits(self):
        self.assertClean("a = train_test_split(X, y, shuffle=False)", "BL005")
        self.assertClean("cv = TimeSeriesSplit(n_splits=5)", "BL005")
        self.assertClean("a = X[:100], X[100:]", "BL005")

    def test_plain_kfold(self):
        self.assertFlags("cv = KFold(n_splits=5)", "BL014")
        self.assertFlags("cv = GroupKFold(n_splits=3)", "BL014")
        self.assertClean("cv = TimeSeriesSplit(n_splits=5)", "BL014")

    def test_fit_before_split(self):
        code = "scaler = StandardScaler()\nXs = scaler.fit_transform(X)\nX_tr, X_te = train_test_split(Xs, y, shuffle=False)"
        self.assertFlags(code, "BL006")
        self.assertFlags("Xs = StandardScaler().fit_transform(X)\na = train_test_split(Xs, y, shuffle=False)", "BL006")
        ok = "scaler = StandardScaler()\nXt = scaler.fit_transform(X_train)\nXv = scaler.transform(X_test)\na = train_test_split(X, y, shuffle=False)"
        self.assertClean(ok, "BL006")
        self.assertClean("scaler = StandardScaler()\nXs = scaler.fit_transform(X)", "BL006")      # no split in the script


class StatisticTests(RuleCase):
    def test_full_sample_zscore(self):
        self.assertFlags("z = (df.close - df.close.mean()) / df.close.std()", "BL007")
        self.assertFlags("z = (x - x.mean()) / x.std()", "BL007")
        self.assertFlags("n = x / x.max()", "BL007")

    def test_trailing_statistics_are_fine(self):
        self.assertClean("z = (x - x.rolling(20).mean()) / x.rolling(20).std()", "BL007")
        self.assertClean("z = (x - x.expanding().mean()) / x.expanding().std()", "BL007")
        self.assertClean("z = (x - x.ewm(span=10).mean()) / x.ewm(span=10).std()", "BL007")
        self.assertClean("z = (a - b.mean()) / c.std()", "BL007")
        self.assertClean("z = (x - train.mean()) / train.std()", "BL007")
        self.assertClean("z = (x - x_train.mean()) / x_train.std()", "BL007")
        self.assertClean("m = x.mean()", "BL007")

    def test_one_report_per_line(self):
        self.assertEqual(ids("z = (x - x.mean()) / x.std()").count("BL007"), 1)

    def test_full_sample_buckets(self):
        self.assertFlags("b = pd.qcut(df.score, 5)", "BL013")
        self.assertFlags("r = df.score.rank(pct=True)", "BL013")
        self.assertClean("r = df.score.rank()", "BL013")
        self.assertClean("r = df.groupby('date').score.rank(pct=True)", "BL013")


class DataHandlingTests(RuleCase):
    def test_backward_fill(self):
        self.assertFlags("x = s.bfill()", "BL008")
        self.assertFlags("x = s.backfill()", "BL008")
        self.assertFlags("x = s.fillna(method='bfill')", "BL008")
        self.assertFlags("x = s.fillna(method='nearest')", "BL008")
        self.assertFlags("x = s.interpolate()", "BL008")
        self.assertClean("x = s.ffill()", "BL008")
        self.assertClean("x = s.fillna(method='ffill')", "BL008")
        self.assertClean("x = s.fillna(0)", "BL008")
        self.assertClean("x = s.interpolate(limit_direction='forward')", "BL008")

    def test_merge_asof(self):
        self.assertFlags("m = pd.merge_asof(a, b, on='t', direction='forward')", "BL009")
        self.assertFlags("m = pd.merge_asof(a, b, on='t', direction='nearest')", "BL009")
        self.assertClean("m = pd.merge_asof(a, b, on='t')", "BL009")
        self.assertClean("m = pd.merge_asof(a, b, on='t', direction='backward')", "BL009")

    def test_survivorship_dropna(self):
        self.assertFlags("p = prices.dropna(axis=1)", "BL010")
        self.assertFlags("p = prices.dropna(axis='columns', how='any')", "BL010")
        self.assertFlags("p = prices.dropna(1)", "BL010")
        self.assertClean("p = prices.dropna()", "BL010")
        self.assertClean("p = prices.dropna(axis=0)", "BL010")
        self.assertClean("p = prices.dropna(how='all')", "BL010")


class SuppressionTests(RuleCase):
    def test_inline_ignore(self):
        self.assertClean("y = s.shift(-1)  # backtestlint: ignore")
        self.assertClean("y = s.shift(-1)  # backtestlint: ignore[BL001]")
        self.assertClean("y = s.shift(-1)  # backtestlint: ignore[BL002, BL001]")
        self.assertFlags("y = s.shift(-1)  # backtestlint: ignore[BL002]", "BL001")
        self.assertFlags("y = s.shift(-1)  # some other comment", "BL001")

    def test_ignore_only_affects_its_line(self):
        self.assertFlags("a = s.shift(-1)  # backtestlint: ignore\nb = s.shift(-2)", "BL001")

    def test_ignore_file(self):
        self.assertEqual(ids("# backtestlint: ignore-file\ny = s.shift(-1)\nz = s.bfill()"), [])


class RobustnessTests(RuleCase):
    def test_syntax_errors_raise_syntax_error(self):
        with self.assertRaises(SyntaxError):
            lint_source("def broken(:\n", "bad.py")

    def test_empty_and_comment_only(self):
        self.assertEqual(ids(""), [])
        self.assertEqual(ids("# nothing here\n"), [])

    def test_odd_but_valid_syntax_does_not_crash(self):
        code = "x = [i for i in range(3)]\nf = lambda a: a.shift(-1)\nasync def g():\n    await h(df.iloc[-1])\nwith open('f') as fh:\n    pass\n@dec\nclass A:\n    y: int = s.shift(-1)\n"
        self.assertIn("BL001", ids(code))

    def test_positions_are_reported(self):
        [f] = lint_source("a = 1\nb = s.shift(-1)\n")
        self.assertEqual((f.line, f.col), (2, 4))
        self.assertEqual(f.to_dict()["name"], "negative-shift")


CLEAN_WALK_FORWARD = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def features(df):
    out = pd.DataFrame(index=df.index)
    out["mom"] = df.close.pct_change(20)
    out["vol"] = df.close.pct_change().rolling(20).std()
    out["z"] = (df.close - df.close.rolling(60).mean()) / df.close.rolling(60).std()
    return out.ffill()


def run(df, model):
    X = features(df).dropna()
    y = df.close.pct_change().shift(-1).loc[X.index]   # backtestlint: ignore[BL001]  (label)
    pnl = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X.iloc[train_idx])
        model.fit(Xtr, y.iloc[train_idx])
        pred = model.predict(scaler.transform(X.iloc[test_idx]))
        position = pd.Series(np.sign(pred), index=X.index[test_idx]).shift(1)
        pnl.append(position * df.close.pct_change().loc[X.index[test_idx]])
    return pd.concat(pnl)
'''


class FalsePositiveGuard(RuleCase):
    def test_a_careful_walk_forward_backtest_is_clean(self):
        self.assertEqual(ids(CLEAN_WALK_FORWARD), [])


if __name__ == "__main__":
    unittest.main()
