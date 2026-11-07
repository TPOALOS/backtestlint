"""A deliberately flawed toy backtest. Every marked line leaks information from the future.

Run:  python3 -m backtestlint examples/leaky_backtest.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    f["trend"] = prices.close.rolling(20, center=True).mean()                      # centered window
    f["zscore"] = (prices.close - prices.close.mean()) / prices.close.std()        # whole-sample statistics
    f["next_ret"] = prices.close.pct_change().shift(-1)                            # tomorrow's return as a feature
    return f.fillna(method="bfill")                                                # next known value fills the gap


def run(prices: pd.DataFrame, model):
    X = StandardScaler().fit_transform(build_features(prices).dropna())            # scaler sees the test period
    y = (prices.close.pct_change().shift(-1) > 0).iloc[: len(X)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)       # random split of a time series
    model.fit(X_train, y_train)
    signal = pd.Series(model.predict(X_test))
    returns = prices.close.pct_change().iloc[-len(signal):].reset_index(drop=True)
    return (signal * returns).cumsum()                                             # signal and return from the same bar
