import pandas as pd
import numpy as np
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from polygon import RESTClient
from config import TICKERS, START_DATE, END_DATE, SENTIMENT_LOOKBACK_DAYS

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "your_key_here")

def load_finbert():
    print("  Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model      = model.to(device)
    model.eval()
    print(f"  FinBERT loaded on {device}")
    return tokenizer, model, device

def score_headlines(headlines, tokenizer, model, device, batch_size=32):
    scores    = []
    label_map = {0: 1.0, 1: -1.0, 2: 0.0}
    for i in range(0, len(headlines), batch_size):
        batch  = headlines[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        scores.extend([label_map[p] for p in preds])
    return scores

def fetch_news_with_retry(client, ticker, max_articles=5000):
    articles = []
    count    = 0
    try:
        for article in client.list_ticker_news(
            ticker,
            published_utc_gte=START_DATE,
            published_utc_lte=END_DATE,
            order="asc",
            limit=50,
        ):
            articles.append(article)
            count += 1
            if count % 50 == 0:
                print(f"    {count} articles fetched, pausing...")
                time.sleep(12)
            if count >= max_articles:
                break
    except Exception as e:
        print(f"  Warning: stopped early for {ticker}: {e}")
    return articles

def parse_article_date(published_utc) -> pd.Timestamp:
    """
    Robustly parses Massive API timestamps into timezone-naive dates.
    Massive returns ISO 8601 UTC strings like '2023-01-15T14:32:00Z'.
    We must strip timezone info to match the price DataFrame's naive index.
    """
    try:
        ts = pd.Timestamp(published_utc)
        # If timezone-aware, convert to UTC then strip tz info
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.normalize()  # floor to midnight
    except Exception:
        return None

def build_sentiment_series(ticker, tokenizer, model, device):
    cache_path = os.path.join("data", f"{ticker}_sentiment.csv")

    if os.path.exists(cache_path):
        print(f"  Loading {ticker} sentiment from cache...")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        # Verify cache has actual signal — if all zeros, re-fetch
        if (df["sentiment"] != 0).sum() == 0:
            print(f"  Cache has no signal — re-fetching {ticker}...")
            os.remove(cache_path)
        else:
            return df["sentiment"]

    client   = RESTClient(api_key=MASSIVE_API_KEY)
    raw      = fetch_news_with_retry(client, ticker)
    full_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    # Ensure index is timezone-naive
    full_idx = full_idx.tz_localize(None)

    if not raw:
        print(f"  No articles for {ticker} — using neutral")
        series = pd.Series(0.0, index=full_idx, name="sentiment")
        series.index.name = "Date"
        series.to_csv(cache_path, header=True)
        return series

    records = []
    for article in raw:
        try:
            pub_date = parse_article_date(article.published_utc)
            title    = article.title or ""
            if pub_date is not None and title:
                records.append({"date": pub_date, "title": title})
        except Exception:
            continue

    if not records:
        print(f"  No parseable articles for {ticker}")
        series = pd.Series(0.0, index=full_idx, name="sentiment")
        series.index.name = "Date"
        series.to_csv(cache_path, header=True)
        return series

    print(f"  Scoring {len(records)} headlines with FinBERT...")
    titles = [r["title"] for r in records]
    scores = score_headlines(titles, tokenizer, model, device)

    df          = pd.DataFrame(records)
    df["score"] = scores
    df["date"]  = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df          = df.set_index("date")
    df          = df.sort_index()

    # Diagnostic — show score distribution before aggregating
    non_neutral = (df["score"] != 0).sum()
    print(f"  Score distribution: "
          f"positive={( df['score']  > 0).sum()} "
          f"negative={(df['score']  < 0).sum()} "
          f"neutral={(df['score'] == 0).sum()} "
          f"total={len(df)}")

    # Daily average — multiple articles per day get averaged
    daily = df["score"].resample("D").mean()

    # Reindex to full business day range — timezone-naive on both sides
    daily = daily.reindex(full_idx)

    # Rolling average to smooth noise
    daily = daily.rolling(SENTIMENT_LOOKBACK_DAYS, min_periods=1).mean()
    daily = daily.fillna(0.0)
    daily.index.name = "Date"
    daily.name       = "sentiment"

    # Final diagnostic
    non_zero = (daily != 0).sum()
    print(f"  {ticker}: {non_zero}/{len(daily)} days have signal "
          f"| mean={daily[daily!=0].mean():.4f}")

    daily.to_csv(cache_path, header=True)
    return daily

def load_all_sentiment():
    tokenizer, model, device = load_finbert()
    sentiment = {}
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        cache_path = os.path.join("data", f"{ticker}_sentiment.csv")
        if os.path.exists(cache_path):
            # Check if cached data has actual signal
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            if (df["sentiment"] != 0).sum() > 0:
                print(f"  Loading {ticker} sentiment from cache...")
                sentiment[ticker] = df["sentiment"]
                continue
            else:
                print(f"  Cache empty — re-fetching {ticker}...")
                os.remove(cache_path)

        sentiment[ticker] = build_sentiment_series(
            ticker, tokenizer, model, device)

        if ticker != TICKERS[-1]:
            print(f"  Waiting 60s before next ticker...")
            time.sleep(60)

    return sentiment

if __name__ == "__main__":
    for ticker in TICKERS:
        path = os.path.join("data", f"{ticker}_sentiment.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if (df["sentiment"] != 0).sum() == 0:
                print(f"Removing empty cache for {ticker}")
                os.remove(path)

    print("Building sentiment series with Massive + FinBERT...")
    tokenizer, model, device = load_finbert()
    for ticker in TICKERS:
        print(f"\n{ticker}:")
        s = build_sentiment_series(ticker, tokenizer, model, device)
        non_zero = (s != 0).sum()
        print(f"  Result: {non_zero} days with signal")
        if ticker != TICKERS[-1]:
            time.sleep(60)