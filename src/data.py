from __future__ import annotations

from typing import List

import pandas as pd
import yfinance as yf


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_tickers(csv_fallback: str) -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia, fallback to CSV on failure."""

    try:
        tables = pd.read_html(WIKI_URL)
        sp500_table = tables[0]
        tickers = sp500_table["Symbol"].astype(str).str.replace(".", "-", regex=False)
        return sorted(tickers.unique().tolist())
    except Exception:
        fallback = pd.read_csv(csv_fallback)
        return sorted(fallback["ticker"].astype(str).unique().tolist())


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.loc[:, (slice(None), "Close")]
        prices.columns = prices.columns.get_level_values(0)
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.sort_index()


def download_spy(start: str, end: str) -> pd.Series:
    """Download SPY adjusted close prices."""

    data = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    return data["Close"].sort_index()
