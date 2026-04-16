import yfinance as yf
import pandas as pd
import os
from config import TICKERS, START_DATE, END_DATE

def fetch_ticker(ticker: str) -> pd.DataFrame:
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)
    df["Ticker"] = ticker
    df.dropna(inplace=True)
    return df

def load_all_tickers(save_csv=True) -> dict:
    all_data = {}
    for ticker in TICKERS:
        cache_path = os.path.join("data", f"{ticker}.csv")
        if os.path.exists(cache_path):
            print(f"  Loading {ticker} from cache...")
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        else:
            df = fetch_ticker(ticker)
            if save_csv:
                df.to_csv(cache_path)
        all_data[ticker] = df
    return all_data

if __name__ == "__main__":
    print("Downloading tickers...")
    data = load_all_tickers()
    for t, df in data.items():
        if len(df) == 0:
            print(f"{t}: FAILED — empty DataFrame")
        else:
            print(f"{t}: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")