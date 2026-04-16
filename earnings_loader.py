import pandas as pd
import numpy as np
import yfinance as yf
import os
from config import TICKERS

def fetch_earnings(ticker: str) -> pd.DataFrame:
    cache_path = os.path.join("data", f"{ticker}_earnings.csv")

    if os.path.exists(cache_path):
        print(f"  Loading {ticker} earnings from cache...")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    print(f"  Fetching {ticker} earnings via yfinance...")
    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates — this gives us quarterly report dates
        # and whether EPS beat or missed
        earnings = stock.earnings_dates
        if earnings is None or earnings.empty:
            print(f"  No earnings data for {ticker}")
            return pd.DataFrame()

        # earnings_dates has columns: EPS Estimate, Reported EPS, Surprise(%)
        earnings = earnings.reset_index()
        earnings.columns = [c.strip() for c in earnings.columns]

        # Rename and clean
        date_col = earnings.columns[0]
        earnings["date"] = pd.to_datetime(earnings[date_col]).dt.tz_localize(None).dt.normalize()

        # Compute surprise from Surprise(%) column if available
        if "Surprise(%)" in earnings.columns:
            earnings["eps_surprise"] = earnings["Surprise(%)"].fillna(0) / 100
            earnings["eps_surprise"] = earnings["eps_surprise"].clip(-2, 2)
        elif "Reported EPS" in earnings.columns and "EPS Estimate" in earnings.columns:
            rep = earnings["Reported EPS"].fillna(0)
            est = earnings["EPS Estimate"].fillna(0)
            earnings["eps_surprise"] = np.clip(
                (rep - est) / (est.abs() + 1e-9), -2, 2)
        else:
            earnings["eps_surprise"] = 0.0

        # No revenue data from yfinance earnings_dates
        earnings["rev_surprise"] = 0.0

        result = earnings[["date", "eps_surprise", "rev_surprise"]].copy()
        result = result.dropna(subset=["date"])
        result = result.sort_values("date").reset_index(drop=True)

        result.to_csv(cache_path, index=False)
        print(f"  {ticker}: {len(result)} earnings dates found")
        return result

    except Exception as e:
        print(f"  Warning: could not fetch earnings for {ticker}: {e}")
        return pd.DataFrame()

def build_earnings_features(ticker: str,
                             price_df: pd.DataFrame) -> pd.DataFrame:
    earnings  = fetch_earnings(ticker)
    price_df  = price_df.copy()
    price_idx = pd.DatetimeIndex(price_df.index).tz_localize(None)

    if earnings.empty:
        for col in get_earnings_feature_columns():
            price_df[col] = 0.0
        return price_df

    earnings["date"] = pd.to_datetime(
        earnings["date"]).dt.tz_localize(None)

    eps_surprise_list  = []
    rev_surprise_list  = []
    days_to_list       = []
    days_since_list    = []
    pead_list          = []

    for date in price_idx:
        past   = earnings[earnings["date"] <= date]
        future = earnings[earnings["date"] >  date]

        if past.empty:
            eps_surprise_list.append(0.0)
            rev_surprise_list.append(0.0)
            days_since_list.append(90)
            pead_list.append(0.0)
        else:
            last       = past.iloc[-1]
            days_since = (date - last["date"]).days
            eps_surp   = float(last["eps_surprise"])
            rev_surp   = float(last["rev_surprise"])
            pead       = eps_surp * np.exp(-days_since / 30.0)

            eps_surprise_list.append(eps_surp)
            rev_surprise_list.append(rev_surp)
            days_since_list.append(min(days_since, 120))
            pead_list.append(np.clip(pead, -2.0, 2.0))

        if future.empty:
            days_to_list.append(90)
        else:
            days_to = (future.iloc[0]["date"] - date).days
            days_to_list.append(min(days_to, 120))

    price_df["eps_surprise"]        = eps_surprise_list
    price_df["rev_surprise"]        = rev_surprise_list
    price_df["days_to_earnings"]    = days_to_list
    price_df["days_since_earnings"] = days_since_list
    price_df["pead_signal"]         = pead_list

    return price_df

def get_earnings_feature_columns() -> list:
    return [
        "eps_surprise",
        "rev_surprise",
        "days_to_earnings",
        "days_since_earnings",
        "pead_signal",
    ]

def load_all_earnings(all_data: dict) -> dict:
    result = {}
    for ticker, df in all_data.items():
        print(f"  Building earnings features for {ticker}...")
        result[ticker] = build_earnings_features(ticker, df)
    return result

if __name__ == "__main__":
    from data_loader import load_all_tickers
    all_data = load_all_tickers()
    for ticker, df in all_data.items():
        enriched = build_earnings_features(ticker, df)
        cols     = get_earnings_feature_columns()
        print(f"\n{ticker} — last 8 rows:")
        print(enriched[cols].tail(8).to_string())