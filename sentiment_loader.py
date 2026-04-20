# sentiment_loader.py
# Replaced Polygon + FinBERT with yfinance news + VADER for speed.
# Same output format as before — nothing else in the pipeline needs to change.

import pandas as pd
import numpy as np
import os
from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

from config import TICKERS, START_DATE, SENTIMENT_LOOKBACK_DAYS

analyzer = SentimentIntensityAnalyzer()

def score_headlines(headlines: list) -> list:
    """Score a list of headlines with VADER. Returns a list of compound scores (-1 to 1)."""
    return [analyzer.polarity_scores(h)["compound"] for h in headlines]

def build_sentiment_series(ticker: str) -> pd.Series:
    cache_path = os.path.join("data", f"{ticker}_sentiment.csv")
    full_idx   = pd.date_range(start=START_DATE, end=str(date.today()), freq="B")
    full_idx   = full_idx.tz_localize(None)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        df.index = df.index.tz_localize(None)
        if (df["sentiment"] != 0).sum() > 0:
            print(f"  {ticker}: loaded from cache")
            return df["sentiment"].reindex(full_idx).ffill().fillna(0)

    print(f"  {ticker}: fetching news from yfinance...")
    try:
        stock    = yf.Ticker(ticker)
        news     = stock.news
    except Exception as e:
        print(f"  {ticker}: could not fetch news — {e}")
        news = []

    if not news:
        print(f"  {ticker}: no news found, using neutral sentiment")
        series = pd.Series(0.0, index=full_idx, name="sentiment")
        series.index.name = "Date"
        series.to_csv(cache_path, header=True)
        return series

    records = []
    for article in news:
        try:
            content  = article.get("content", {})
            title    = content.get("title", "")
            pub_date = pd.Timestamp(content["pubDate"]).tz_localize(None).normalize()
            if title:
                records.append({"date": pub_date, "title": title})
        except Exception:
            continue

    if not records:
        print(f"  {ticker}: no parseable articles")
        series = pd.Series(0.0, index=full_idx, name="sentiment")
        series.index.name = "Date"
        series.to_csv(cache_path, header=True)
        return series

    print(f"  {ticker}: scoring {len(records)} headlines with VADER...")
    titles = [r["title"] for r in records]
    scores = score_headlines(titles)

    df          = pd.DataFrame(records)
    df["score"] = scores
    df["date"]  = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df          = df.set_index("date").sort_index()

    daily = df["score"].resample("D").mean()
    daily = daily.reindex(full_idx)
    daily = daily.rolling(SENTIMENT_LOOKBACK_DAYS, min_periods=1).mean()
    daily = daily.fillna(0.0)
    daily.index.name = "Date"
    daily.name       = "sentiment"

    non_zero = (daily != 0).sum()
    print(f"  {ticker}: {non_zero} days with signal | mean={daily[daily != 0].mean():.4f}")

    daily.to_csv(cache_path, header=True)
    return daily

def load_all_sentiment() -> dict:
    print("Loading sentiment (yfinance + VADER)...")
    sentiment = {}
    for ticker in TICKERS:
        sentiment[ticker] = build_sentiment_series(ticker)
    print("Sentiment done.")
    return sentiment

if __name__ == "__main__":
    # Clear old cache so we get fresh VADER-scored data
    for ticker in TICKERS:
        path = os.path.join("data", f"{ticker}_sentiment.csv")
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleared cache for {ticker}")

    load_all_sentiment()