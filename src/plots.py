from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import compute_drawdown


def plot_equity_curve(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str,
    title: str,
) -> None:
    """Plot strategy vs benchmark equity curves."""

    strategy_curve = (1.0 + strategy_returns).cumprod()
    benchmark_curve = (1.0 + benchmark_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(strategy_curve.index, strategy_curve, label="Strategy")
    plt.plot(benchmark_curve.index, benchmark_curve, label="SPY", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(strategy_returns: pd.Series, output_path: str, title: str) -> None:
    """Plot drawdown curve for the strategy."""

    equity_curve = (1.0 + strategy_returns).cumprod()
    drawdown = compute_drawdown(equity_curve)

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown, color="tab:red")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rolling_sharpe(
    rolling_sharpe: pd.Series,
    output_path: str,
    title: str,
) -> None:
    """Plot rolling 12-month Sharpe ratio."""

    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color="tab:blue")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe (12M)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_turnover(turnover: pd.Series, output_path: str, title: str) -> None:
    """Plot turnover time series."""

    plt.figure(figsize=(10, 4))
    plt.plot(turnover.index, turnover, color="tab:purple")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
