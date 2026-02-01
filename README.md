# Empirical Analysis of Equity Factor Strategies Under Market Regimes (Student Research Project)

## Project Overview
This repository is a **research project** that studies whether simple, long-only equity factor strategies deliver superior risk-adjusted returns versus the market (SPY). The project focuses on **Momentum** and **Low Volatility** factors within the **S&P 500** universe and analyzes when these strategies underperform across different market regimes.

## Factors and Definitions
All signals are computed using **daily adjusted close prices** (via `yfinance`). Strategies rebalance **monthly on the first trading day** and hold **equal-weighted** portfolios.

### Momentum (12–1)
For each stock, compute cumulative return from **t-252 to t-21** trading days:

```
Momentum_i = (Price_i[t-21] / Price_i[t-252]) - 1
```

Rank stocks **descending** and go **long top 10%**.

### Low Volatility
Compute trailing **63 trading-day** realized volatility of daily returns:

```
Vol_i = StdDev( DailyReturns_i over past 63 days )
```

Rank stocks **ascending** and go **long top 10% (lowest volatility)**.

### Combined (Optional)
Average normalized ranks of momentum and low-vol signals, then select top 10%.

## Rebalancing Rules
- **Frequency:** Monthly (first trading day).
- **Weights:** Equal-weighted across selected stocks.
- **Missing Data:** Stocks without enough lookback history at each rebalance are dropped.

## Transaction Cost Model
Simple turnover-based model applied at rebalance:

```
Turnover = sum(|w_new - w_old|)
TransactionCost = Turnover * (cost_bps / 10000)
```

Set `--cost_bps 0` to disable costs.

## How to Run
Install dependencies and run from repo root:

```
pip install -r requirements.txt
```

Example CLI usage:

```
python -m src.backtest --start 2013-01-01 --end 2025-12-31 --strategy momentum --cost_bps 10
python -m src.backtest --strategy low_vol
python -m src.backtest --strategy combo --cost_bps 0
```

Supported strategies: `momentum`, `low_vol`, `combo`.

## Output Files Produced
Running the CLI creates:

- `reports/summary.md` — research note with metrics and regime analysis.
- `reports/<strategy>_daily_returns.csv` — daily strategy returns.
- `reports/<strategy>_turnover.csv` — turnover series.
- `reports/figures/<strategy>_equity_curve.png`
- `reports/figures/<strategy>_drawdown.png`
- `reports/figures/<strategy>_rolling_sharpe.png`
- `reports/figures/<strategy>_turnover.png`

## Limitations
- **Survivorship bias:** the S&P 500 list is current, not historical.
- **Free data quality:** `yfinance` may have missing data or irregularities.
- **Simplified costs:** turnover-based model is a proxy.
- **Long-only assumption:** no shorts or leverage.

## Project Structure
```
repo/
  README.md
  requirements.txt
  src/
    __init__.py
    data.py
    factors.py
    backtest.py
    metrics.py
    plots.py
    utils.py
  reports/
    summary.md
    figures/
  data/
    sp500_tickers.csv
```
