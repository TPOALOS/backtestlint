"""Rule catalogue: ids, severities, explanations."""

SEVERITIES = ("HIGH", "MEDIUM", "LOW")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")


class Rule:
    def __init__(self, rid, name, severity, summary, why, fix):
        self.id, self.name, self.severity, self.summary, self.why, self.fix = rid, name, severity, summary, why, fix


RULES = {r.id: r for r in [
    Rule("BL001", "negative-shift", "HIGH", "shift() with a negative period reads future rows",
         "shift(-n) moves value from n rows ahead into the current row. If it reaches a feature, signal or filter, the backtest sees the future.",
         "Use positive shifts for features. If this builds a prediction label, keep it out of features and silence the line with `# backtestlint: ignore[BL001]`."),
    Rule("BL002", "centered-window", "HIGH", "rolling(center=True) uses data after the current row",
         "A centered window averages values before and after each row, so every value depends on the future.",
         "Use the default trailing window (center=False)."),
    Rule("BL003", "future-index-in-loop", "MEDIUM", "loop index plus a positive offset reads a later row",
         "Inside a bar-by-bar loop, data[i + 1] is information from the next bar.",
         "Only read later bars for fills after the decision is made (for example the next open), never for signals."),
    Rule("BL004", "same-bar-execution", "MEDIUM", "signal multiplied by returns without shifting the signal",
         "If today's signal is computed from today's close and multiplied by today's return, the strategy trades at a price it could not have known.",
         "Shift the signal by at least one bar: position = signal.shift(1)."),
    Rule("BL005", "shuffled-split", "HIGH", "random train/test split of ordered data",
         "Shuffling time series puts future rows in the training set, so test scores are inflated.",
         "Split by time (shuffle=False, or TimeSeriesSplit / walk-forward) and keep a gap if labels overlap."),
    Rule("BL006", "fit-before-split", "MEDIUM", "scaler/transformer fitted before the data is split",
         "Fitting normalisers, PCA or imputers on all rows leaks test-period statistics into training.",
         "Fit on the training slice only, then transform the test slice."),
    Rule("BL007", "full-sample-statistic", "MEDIUM", "feature normalised with a whole-sample statistic",
         "(x - x.mean()) / x.std() uses the mean and deviation of the entire history, including the future.",
         "Use rolling()/expanding() statistics, or statistics from the training window only."),
    Rule("BL008", "backward-fill", "HIGH", "backward fill or nearest interpolation copies future values into the past",
         "bfill() and method='bfill'/'nearest' fill gaps with the next known value.",
         "Use forward fill (ffill) or drop the rows."),
    Rule("BL009", "future-join", "HIGH", "merge_asof looks forward or nearest",
         "direction='forward' or 'nearest' attaches later observations to earlier rows.",
         "Use direction='backward' (the default) and allow_exact_matches=False where the data is published after the timestamp."),
    Rule("BL010", "survivorship-dropna", "MEDIUM", "dropna(axis=1) removes assets with missing history",
         "Dropping columns with gaps silently deletes delisted or late-listed assets, so only survivors remain.",
         "Keep the full point-in-time universe and handle missing data explicitly."),
    Rule("BL011", "future-return", "MEDIUM", "pct_change/diff with a negative period computes forward changes",
         "Forward returns are fine as labels but are look-ahead if used as features or filters.",
         "Keep forward changes out of features; silence the line if this is a label."),
    Rule("BL012", "last-row-in-loop", "HIGH", "iloc[-1] / [-1] inside a bar loop is the final row of the whole dataset",
         "While looping over rows, [-1] refers to the last bar of the data, not the current bar.",
         "Index with the loop variable (iloc[i]) or iterate with expanding windows."),
    Rule("BL013", "full-sample-rank", "LOW", "qcut/rank over the whole sample",
         "Quantile buckets and percentile ranks computed on all rows depend on future data.",
         "Rank cross-sectionally per date or use rolling/expanding ranks."),
    Rule("BL014", "overlapping-cv", "MEDIUM", "plain K-fold on ordered data",
         "K-fold trains on later folds and tests on earlier ones; overlapping labels leak across folds.",
         "Use TimeSeriesSplit or purged/embargoed splits."),
]}
