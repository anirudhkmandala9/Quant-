from __future__ import annotations

import pandas as pd


def momentum_signal(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Compute 12-1 momentum signal for each ticker at a given date."""

    shifted_21 = prices.shift(21)
    shifted_252 = prices.shift(252)
    momentum = shifted_21 / shifted_252 - 1.0
    if as_of not in momentum.index:
        raise KeyError("as_of date not in price index")
    return momentum.loc[as_of]


def low_vol_signal(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Compute 63-day trailing volatility signal for each ticker at a given date."""

    daily_returns = prices.pct_change()
    rolling_vol = daily_returns.rolling(window=63).std()
    if as_of not in rolling_vol.index:
        raise KeyError("as_of date not in price index")
    return rolling_vol.loc[as_of]


def combo_signal(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Compute combined normalized rank of momentum and low-vol signals."""

    momentum = momentum_signal(prices, as_of)
    low_vol = low_vol_signal(prices, as_of)
    momentum_rank = momentum.rank(pct=True, ascending=False)
    low_vol_rank = low_vol.rank(pct=True, ascending=True)
    combined = (momentum_rank + low_vol_rank) / 2.0
    return combined
