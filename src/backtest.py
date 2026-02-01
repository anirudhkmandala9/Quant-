from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src import data as data_loader
from src import factors
from src import metrics
from src import plots
from src.utils import (
    BacktestConfig,
    dataframe_to_markdown,
    ensure_directory,
    first_trading_days,
)

STRATEGIES = {"momentum", "low_vol", "combo"}


def parse_args() -> BacktestConfig:
    """Parse CLI arguments for the backtest."""

    parser = argparse.ArgumentParser(description="Equity factor backtest")
    parser.add_argument("--start", type=str, default="2013-01-01")
    parser.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--strategy", type=str, required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--cost_bps", type=float, default=10.0)

    args = parser.parse_args()
    return BacktestConfig(
        start=args.start,
        end=args.end,
        strategy=args.strategy,
        cost_bps=args.cost_bps,
    )


def build_weights(
    prices: pd.DataFrame,
    strategy: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create daily weight matrix and turnover series."""

    rebal_dates = first_trading_days(prices.index)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, data=np.nan)
    turnover = pd.Series(index=rebal_dates, dtype=float)
    prev_weights = pd.Series(0.0, index=prices.columns)

    for date in rebal_dates:
        if strategy == "momentum":
            signal = factors.momentum_signal(prices, date)
            signal = signal.dropna().sort_values(ascending=False)
        elif strategy == "low_vol":
            signal = factors.low_vol_signal(prices, date)
            signal = signal.dropna().sort_values(ascending=True)
        else:
            signal = factors.combo_signal(prices, date)
            signal = signal.dropna().sort_values(ascending=False)

        if signal.empty:
            continue

        top_n = max(1, int(len(signal) * 0.10))
        selected = signal.iloc[:top_n].index
        new_weights = pd.Series(0.0, index=prices.columns)
        new_weights.loc[selected] = 1.0 / len(selected)

        weights.loc[date] = new_weights
        turnover.loc[date] = (new_weights - prev_weights).abs().sum()
        prev_weights = new_weights

    weights = weights.ffill().fillna(0.0)
    turnover = turnover.fillna(0.0)
    return weights, turnover


def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    turnover: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Compute daily portfolio returns with turnover-based transaction costs."""

    daily_returns = prices.pct_change().fillna(0.0)
    portfolio_returns = (weights.shift(1).fillna(0.0) * daily_returns).sum(axis=1)

    cost_rate = cost_bps / 10000.0
    if cost_rate > 0:
        cost_series = pd.Series(0.0, index=portfolio_returns.index)
        cost_series.loc[turnover.index] = turnover * cost_rate
        portfolio_returns = portfolio_returns - cost_series

    return portfolio_returns


def save_outputs(
    config: BacktestConfig,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series,
    spy_prices: pd.Series,
) -> None:
    """Save figures, CSVs, and summary report."""

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    ensure_directory(str(figures_dir))

    strategy_name = config.strategy
    strategy_returns.to_csv(reports_dir / f"{strategy_name}_daily_returns.csv", header=True)
    turnover.to_csv(reports_dir / f"{strategy_name}_turnover.csv", header=True)

    plots.plot_equity_curve(
        strategy_returns,
        benchmark_returns,
        str(figures_dir / f"{strategy_name}_equity_curve.png"),
        title=f"{strategy_name.title()} Strategy vs SPY",
    )
    plots.plot_drawdown(
        strategy_returns,
        str(figures_dir / f"{strategy_name}_drawdown.png"),
        title=f"{strategy_name.title()} Strategy Drawdown",
    )

    rolling_sharpe = metrics.rolling_sharpe(strategy_returns)
    plots.plot_rolling_sharpe(
        rolling_sharpe,
        str(figures_dir / f"{strategy_name}_rolling_sharpe.png"),
        title=f"{strategy_name.title()} Rolling 12M Sharpe",
    )

    plots.plot_turnover(
        turnover,
        str(figures_dir / f"{strategy_name}_turnover.png"),
        title=f"{strategy_name.title()} Turnover",
    )

    summary = metrics.performance_summary(strategy_returns, benchmark_returns)
    benchmark_summary = metrics.performance_summary(benchmark_returns, benchmark_returns)
    regime = metrics.regime_analysis(strategy_returns, spy_prices)
    monthly_table = metrics.monthly_return_table(strategy_returns)

    summary_path = reports_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("# Research Note: Equity Factor Strategies Under Market Regimes\n\n")
        file.write("## Methods Overview\n")
        file.write(
            "This project tests long-only momentum and low-volatility strategies "
            "in the S&P 500. Portfolios rebalance monthly on the first trading day, "
            "use equal weights, and apply a turnover-based transaction cost model.\n\n"
        )
        file.write("## Key Performance Metrics\n\n")
        file.write("| Metric | Strategy | SPY |\n")
        file.write("| --- | --- | --- |\n")
        for key in summary:
            file.write(
                f"| {key} | {summary[key]:.4f} | {benchmark_summary[key]:.4f} |\n"
            )
        file.write("\n## Interpretation\n")
        file.write(
            "Momentum and low-volatility strategies can produce different risk/return "
            "profiles than the market. The metrics above highlight whether the factor "
            "delivered higher risk-adjusted returns, but they also show periods of drawdowns "
            "and potential underperformance.\n\n"
        )
        file.write("## Regime Analysis\n\n")
        file.write(
            "Regimes are defined using SPY: high/low volatility via the median of 63-day "
            "rolling volatility, and drawdown regimes based on whether SPY drawdown is worse "
            "than -10%.\n\n"
        )
        file.write(dataframe_to_markdown(regime))
        file.write("\n\n")
        file.write("## Monthly Return Table\n\n")
        file.write(dataframe_to_markdown(monthly_table))
        file.write("\n\n")
        file.write("## Suggested Next Steps\n")
        file.write(
            "- Add sector-neutral constraints to reduce unintended bets.\n"
            "- Use a survivorship-bias-free universe for more realistic results.\n"
            "- Extend the transaction cost model with bid/ask spreads and market impact.\n"
        )


def run_backtest(config: BacktestConfig) -> None:
    """Run the end-to-end backtest and save outputs."""

    tickers = data_loader.fetch_sp500_tickers("data/sp500_tickers.csv")
    prices = data_loader.download_prices(tickers, config.start, config.end)
    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError("No price data downloaded. Check ticker list or date range.")

    weights, turnover = build_weights(prices, config.strategy)
    strategy_returns = compute_portfolio_returns(
        prices, weights, turnover, config.cost_bps
    )

    spy_prices = data_loader.download_spy(config.start, config.end)
    spy_returns = spy_prices.pct_change().fillna(0.0)

    aligned = pd.concat([strategy_returns, spy_returns], axis=1).dropna()
    strategy_returns = aligned.iloc[:, 0]
    spy_returns = aligned.iloc[:, 1]

    save_outputs(config, strategy_returns, spy_returns, turnover, spy_prices)


def main() -> None:
    """CLI entry point."""

    config = parse_args()
    run_backtest(config)


if __name__ == "__main__":
    main()
