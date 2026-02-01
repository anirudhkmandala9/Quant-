from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the backtest run."""

    start: str
    end: str
    strategy: str
    cost_bps: float


def ensure_directory(path: str) -> None:
    """Create a directory if it does not exist."""

    import os

    os.makedirs(path, exist_ok=True)


def first_trading_days(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Return the first trading day of each month from a DatetimeIndex."""

    if index.empty:
        return []
    monthly_groups = index.to_series().groupby(index.to_period("M"))
    return [group.iloc[0] for _, group in monthly_groups]


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown from an equity curve."""

    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Aggregate daily returns into monthly returns."""

    return daily_returns.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)


def align_series(series_list: Iterable[pd.Series]) -> List[pd.Series]:
    """Align multiple series on their shared dates and drop missing values."""

    aligned = pd.concat(series_list, axis=1).dropna()
    return [aligned.iloc[:, i] for i in range(aligned.shape[1])]


def annualize_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute CAGR from daily returns."""

    if daily_returns.empty:
        return np.nan
    cumulative = (1.0 + daily_returns).prod()
    years = daily_returns.shape[0] / periods_per_year
    if years == 0:
        return np.nan
    return cumulative ** (1.0 / years) - 1.0


def annualize_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized volatility from daily returns."""

    return daily_returns.std() * np.sqrt(periods_per_year)


def annualized_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio assuming rf=0."""

    vol = daily_returns.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return daily_returns.mean() / vol * np.sqrt(periods_per_year)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a minimal markdown table without extra dependencies."""

    def format_value(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.4f}"
        return str(value)

    table = df.reset_index()
    headers = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        row_values = [format_value(value) for value in row]
        lines.append("| " + " | ".join(row_values) + " |")
    return "\n".join(lines)
