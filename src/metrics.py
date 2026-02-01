from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    align_series,
    annualize_return,
    annualize_volatility,
    annualized_sharpe,
    compute_drawdown,
    to_monthly_returns,
)


def performance_summary(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """Compute headline performance metrics."""

    aligned_strategy, aligned_benchmark = align_series(
        [daily_returns, benchmark_returns]
    )
    equity_curve = (1.0 + aligned_strategy).cumprod()
    drawdown = compute_drawdown(equity_curve)
    beta, corr = regression_beta_corr(aligned_strategy, aligned_benchmark)
    return {
        "CAGR": annualize_return(aligned_strategy),
        "Annualized Volatility": annualize_volatility(aligned_strategy),
        "Sharpe Ratio": annualized_sharpe(aligned_strategy),
        "Max Drawdown": drawdown.min(),
        "Beta vs SPY": beta,
        "Correlation vs SPY": corr,
    }


def regression_beta_corr(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> Tuple[float, float]:
    """Compute beta and correlation vs benchmark using linear regression."""

    aligned_strategy, aligned_benchmark = align_series(
        [strategy_returns, benchmark_returns]
    )
    if aligned_strategy.empty:
        return np.nan, np.nan
    covariance = np.cov(aligned_strategy, aligned_benchmark)[0, 1]
    variance = np.var(aligned_benchmark)
    beta = covariance / variance if variance != 0 else np.nan
    corr = np.corrcoef(aligned_strategy, aligned_benchmark)[0, 1]
    return beta, corr


def rolling_sharpe(daily_returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""

    def sharpe(series: pd.Series) -> float:
        return annualized_sharpe(series)

    return daily_returns.rolling(window=window).apply(sharpe, raw=False)


def monthly_return_table(daily_returns: pd.Series) -> pd.DataFrame:
    """Create a year x month table of monthly returns."""

    monthly_returns = to_monthly_returns(daily_returns)
    table = (
        monthly_returns.to_frame(name="return")
        .assign(year=lambda df: df.index.year, month=lambda df: df.index.month)
        .pivot(index="year", columns="month", values="return")
        .sort_index()
    )
    return table


def regime_analysis(
    strategy_daily_returns: pd.Series,
    benchmark_prices: pd.Series,
) -> pd.DataFrame:
    """Evaluate strategy performance across volatility and drawdown regimes."""

    spy_returns = benchmark_prices.pct_change()
    spy_vol = spy_returns.rolling(window=63).std()
    spy_drawdown = compute_drawdown((1.0 + spy_returns).cumprod())

    monthly_strategy = to_monthly_returns(strategy_daily_returns)
    monthly_vol = spy_vol.resample("M").last()
    monthly_drawdown = spy_drawdown.resample("M").last()

    aligned = pd.concat(
        [monthly_strategy, monthly_vol, monthly_drawdown], axis=1
    ).dropna()
    aligned.columns = ["strategy", "vol", "drawdown"]

    vol_median = aligned["vol"].median()
    aligned["vol_regime"] = np.where(aligned["vol"] >= vol_median, "High Vol", "Low Vol")
    aligned["dd_regime"] = np.where(
        aligned["drawdown"] <= -0.10, "Drawdown <= -10%", "Drawdown > -10%"
    )

    def summarize(group: pd.Series) -> Tuple[float, float]:
        avg_monthly = group.mean()
        annualized_sharpe = annualized_sharpe_from_monthly(group)
        return avg_monthly, annualized_sharpe

    results = []
    for regime_col in ["vol_regime", "dd_regime"]:
        for regime, group in aligned.groupby(regime_col):
            avg_monthly, sharpe_ann = summarize(group["strategy"])
            results.append(
                {
                    "Regime Type": regime_col,
                    "Regime": regime,
                    "Avg Monthly Return": avg_monthly,
                    "Annualized Sharpe": sharpe_ann,
                }
            )

    return pd.DataFrame(results)


def annualized_sharpe_from_monthly(monthly_returns: pd.Series) -> float:
    """Compute annualized Sharpe ratio from monthly returns."""

    if monthly_returns.std() == 0 or monthly_returns.empty:
        return np.nan
    return (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
