import yfinance as yf
import pandas as pd
import numpy as np
import os
from config import START_DATE, END_DATE, MACRO_TICKERS

def fetch_macro() -> pd.DataFrame:
    cache_path = os.path.join("data", "macro.csv")
    if os.path.exists(cache_path):
        print("  Loading macro indicators from cache...")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return df

    print("  Fetching macro indicators...")
    frames = {}
    for name, symbol in MACRO_TICKERS.items():
        raw = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frames[name] = raw["Close"].rename(name)

    macro = pd.concat(frames.values(), axis=1, join="outer")
    macro = macro.sort_index()
    macro = macro.ffill()
    macro = macro.dropna()

    macro["vix_change"]      = macro["vix"].pct_change(1)
    macro["treasury_change"] = macro["treasury"].pct_change(1)
    macro["dollar_change"]   = macro["dollar"].pct_change(1)
    macro["vix_ma10"]        = macro["vix"].rolling(10).mean()
    macro["treasury_ma10"]   = macro["treasury"].rolling(10).mean()
    macro["vix_regime"]      = pd.cut(
        macro["vix"],
        bins=[0, 15, 20, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(float)
    macro["vix_zscore"] = (
        (macro["vix"] - macro["vix"].rolling(30).mean()) /
        (macro["vix"].rolling(30).std() + 1e-9)
    )

    macro = macro.dropna()
    macro.to_csv(cache_path)
    print(f"  Macro: {len(macro)} rows | {macro.index[0].date()} to {macro.index[-1].date()}")
    return macro

def get_macro_feature_columns() -> list:
    return [
        "vix", "vix_change", "vix_ma10", "vix_regime", "vix_zscore",
        "treasury", "treasury_change", "treasury_ma10",
        "dollar", "dollar_change",
    ]

if __name__ == "__main__":
    df = fetch_macro()
    print(df.tail())
    print(f"\nFeatures: {get_macro_feature_columns()}")