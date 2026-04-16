# backtest.py
# Holding period analysis — measures how profitable buy signals are
# over extended periods (1 week, 1 month, 1 quarter, 6 months).
#
# This answers the question: if you bought every time XGBoost said
# "up" and held for N days, how did you do vs just holding the stock?
#
# Key metrics produced:
#   - Average return per signal over each holding period
#   - Win rate (% of signals that were profitable at each horizon)
#   - Signal return vs buy-and-hold benchmark
#   - Cumulative P&L curve over time
#   - Quarterly breakdown — which quarters did signals work best?

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_all_tickers
from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from features import add_features, get_feature_columns
from config import TICKERS, TRAIN_RATIO, VAL_RATIO, HOLDING_PERIODS

def get_signal_returns(ticker, df, macro_df=None, sentiment=None):
    """
    For each day in the TEST SET where XGBoost predicted "up",
    calculates the actual forward return over each holding period.

    Returns a DataFrame where each row is one buy signal with columns:
        date           — when the signal fired
        close          — price at signal
        signal         — 1 (buy) or 0 (sell)
        prob_up        — model confidence
        return_5d      — actual return if held 5 days
        return_21d     — actual return if held 21 days
        return_63d     — actual return if held 63 days
        return_126d    — actual return if held 126 days
        quarter        — fiscal quarter (e.g. 2023Q1)
    """
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    model_path  = os.path.join("models", f"{ticker}_xgb.pkl")

    if not os.path.exists(model_path):
        print(f"  No model for {ticker} — run train_classical.py first")
        return None

    sentiment_series = sentiment.get(ticker) if sentiment else None
    df_feat = add_features(df.copy(), macro_df=macro_df,
                           sentiment_series=sentiment_series)
    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment_series is not None
    )
    feat_cols = [c for c in feat_cols if c in df_feat.columns]

    scaler   = joblib.load(scaler_path)
    model    = joblib.load(model_path)
    X_scaled = scaler.transform(df_feat[feat_cols].values)
    probs    = model.predict_proba(X_scaled)[:, 1]

    df_feat["prob_up"] = probs
    df_feat["signal"]  = (probs >= 0.5).astype(int)

    # Restrict to TEST SET only — honest evaluation
    n         = len(df_feat)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_df   = df_feat.iloc[val_end:].copy()
    close_all = df_feat["Close"]  # need full price history for forward returns

    records = []
    for i, (date, row) in enumerate(test_df.iterrows()):
        # Absolute position in the full dataframe
        abs_idx = val_end + i
        record  = {
            "date":    date,
            "close":   row["Close"],
            "signal":  int(row["signal"]),
            "prob_up": row["prob_up"],
            "quarter": f"{date.year}Q{(date.month - 1) // 3 + 1}",
        }

        # Forward returns for each holding period
        for days in HOLDING_PERIODS:
            future_idx = abs_idx + days
            if future_idx < len(close_all):
                future_price    = close_all.iloc[future_idx]
                forward_return  = (future_price - row["Close"]) / row["Close"]
                record[f"return_{days}d"] = forward_return
            else:
                record[f"return_{days}d"] = np.nan

        records.append(record)

    return pd.DataFrame(records)

def holding_period_report(ticker, signals_df):
    """
    Prints a detailed breakdown of signal profitability
    across all holding periods and by quarter.
    """
    if signals_df is None or signals_df.empty:
        return

    buy_signals  = signals_df[signals_df["signal"] == 1]
    sell_signals = signals_df[signals_df["signal"] == 0]
    all_days     = signals_df  # benchmark = hold every day

    print(f"\n{'═'*60}")
    print(f"  HOLDING PERIOD ANALYSIS — {ticker}")
    print(f"{'═'*60}")
    print(f"  Test period: {signals_df['date'].min().date()} "
          f"to {signals_df['date'].max().date()}")
    print(f"  Total test days:  {len(signals_df)}")
    print(f"  Buy signals:      {len(buy_signals)} "
          f"({len(buy_signals)/len(signals_df):.1%} of days)")
    print(f"  Sell signals:     {len(sell_signals)}")

    print(f"\n  {'Period':<12} {'Buy avg':>10} {'Buy win%':>10} "
          f"{'BH avg':>10} {'Edge':>10}")
    print(f"  {'─'*55}")

    for days in HOLDING_PERIODS:
        col = f"return_{days}d"
        if col not in signals_df.columns:
            continue

        buy_returns = buy_signals[col].dropna()
        all_returns = all_days[col].dropna()

        if buy_returns.empty:
            continue

        buy_avg  = buy_returns.mean()
        buy_win  = (buy_returns > 0).mean()
        bh_avg   = all_returns.mean()
        edge     = buy_avg - bh_avg

        period_label = (
            f"{days}d" if days < 21 else
            f"{days//21}mo" if days < 63 else
            f"{days//63}q"  if days < 126 else "6mo"
        )

        print(f"  {period_label:<12} "
              f"{buy_avg:>+10.2%} "
              f"{buy_win:>10.1%} "
              f"{bh_avg:>+10.2%} "
              f"{edge:>+10.2%}")

    # Quarterly breakdown for the longest holding period
    long_period = max(HOLDING_PERIODS)
    col         = f"return_{long_period}d"

    if col in buy_signals.columns:
        print(f"\n  Quarterly breakdown — {long_period}d hold after buy signal")
        print(f"  {'Quarter':<10} {'Signals':>8} {'Avg return':>12} "
              f"{'Win rate':>10} {'Best':>10} {'Worst':>10}")
        print(f"  {'─'*60}")

        for quarter, group in buy_signals.groupby("quarter"):
            returns = group[col].dropna()
            if returns.empty:
                continue
            print(f"  {quarter:<10} "
                  f"{len(returns):>8} "
                  f"{returns.mean():>+12.2%} "
                  f"{(returns > 0).mean():>10.1%} "
                  f"{returns.max():>+10.2%} "
                  f"{returns.min():>+10.2%}")

def plot_holding_period_chart(ticker, signals_df):
    """
    Generates an interactive chart showing:
    - Top: cumulative return of buy-signal strategy vs buy-and-hold
    - Bottom: distribution of returns at each holding period
    """
    if signals_df is None or signals_df.empty:
        return

    buy_signals = signals_df[signals_df["signal"] == 1].copy()
    buy_signals = buy_signals.sort_values("date")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cumulative return — buy signals vs buy-and-hold (1 month hold)",
            "Average return by quarter (1 quarter hold)",
            "Return distribution at each holding period",
            "Win rate by holding period",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # ── Panel 1: Cumulative returns ───────────────────────────────────────────
    # Buy signal strategy: invest on every buy signal, hold 21 days
    col_21 = "return_21d"
    if col_21 in buy_signals.columns:
        buy_cum = (1 + buy_signals[col_21].fillna(0)).cumprod() - 1

        # Buy and hold benchmark over same period
        bh_returns = signals_df[col_21].fillna(0)
        bh_cum     = (1 + bh_returns).cumprod() - 1

        fig.add_trace(go.Scatter(
            x=buy_signals["date"],
            y=buy_cum * 100,
            mode="lines",
            name="Buy signals (1mo hold)",
            line=dict(color="#1D9E75", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=signals_df["date"],
            y=bh_cum * 100,
            mode="lines",
            name="Buy and hold",
            line=dict(color="#888780", width=1.5, dash="dash"),
        ), row=1, col=1)

        fig.add_hline(y=0, line_color="#E24B4A",
                      line_width=0.5, opacity=0.5, row=1, col=1)

    # ── Panel 2: Quarterly average returns ───────────────────────────────────
    col_63 = "return_63d"
    if col_63 in buy_signals.columns:
        quarterly = buy_signals.groupby("quarter")[col_63].mean().dropna()
        colors    = ["#1D9E75" if v >= 0 else "#E24B4A"
                     for v in quarterly.values]

        fig.add_trace(go.Bar(
            x=quarterly.index,
            y=quarterly.values * 100,
            marker_color=colors,
            name="Avg return per quarter",
            showlegend=False,
        ), row=1, col=2)

        fig.add_hline(y=0, line_color="#888780",
                      line_width=0.5, row=1, col=2)

    # ── Panel 3: Return distributions ────────────────────────────────────────
    colors_dist = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"]
    for days, color in zip(HOLDING_PERIODS, colors_dist):
        col = f"return_{days}d"
        if col not in buy_signals.columns:
            continue
        returns = buy_signals[col].dropna() * 100
        label   = (f"{days}d" if days < 21 else
                   f"{days//21}mo" if days < 63 else
                   f"{days//63}q"  if days < 126 else "6mo")
        fig.add_trace(go.Violin(
            y=returns,
            name=label,
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=color,
            opacity=0.4,
        ), row=2, col=1)

    # ── Panel 4: Win rates ────────────────────────────────────────────────────
    win_rates  = []
    period_labels = []
    for days in HOLDING_PERIODS:
        col = f"return_{days}d"
        if col not in buy_signals.columns:
            continue
        wr    = (buy_signals[col].dropna() > 0).mean() * 100
        label = (f"{days}d" if days < 21 else
                 f"{days//21}mo" if days < 63 else
                 f"{days//63}q"  if days < 126 else "6mo")
        win_rates.append(wr)
        period_labels.append(label)

    fig.add_trace(go.Bar(
        x=period_labels,
        y=win_rates,
        marker_color=["#1D9E75" if w >= 50 else "#E24B4A"
                      for w in win_rates],
        name="Win rate %",
        showlegend=False,
        text=[f"{w:.1f}%" for w in win_rates],
        textposition="outside",
    ), row=2, col=2)

    fig.add_hline(y=50, line_dash="dash",
                  line_color="#888780", line_width=1,
                  opacity=0.5, row=2, col=2)

    fig.update_layout(
        height=800,
        title=dict(
            text=f"{ticker} — Holding Period Analysis (test set only)",
            font=dict(size=16)
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="Cumulative return %", row=1, col=1)
    fig.update_yaxes(title_text="Avg return %",        row=1, col=2)
    fig.update_yaxes(title_text="Return %",            row=2, col=1)
    fig.update_yaxes(title_text="Win rate %",
                     range=[0, 100],                   row=2, col=2)

    os.makedirs("charts", exist_ok=True)
    out_path = os.path.join("charts", f"{ticker}_holding_periods.html")
    fig.write_html(out_path)
    print(f"\n  Chart saved: {out_path}")

def run_full_backtest():
    print("Loading data...")
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()

    all_signals = {}

    for ticker, df in all_data.items():
        print(f"\nAnalyzing {ticker}...")
        signals_df = get_signal_returns(
            ticker, df, macro_df=macro_df, sentiment=sentiment)

        if signals_df is not None:
            all_signals[ticker] = signals_df
            holding_period_report(ticker, signals_df)
            plot_holding_period_chart(ticker, signals_df)

    # Cross-ticker summary
    print(f"\n{'═'*60}")
    print("  CROSS-TICKER SUMMARY — 1 quarter hold (63 days)")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<8} {'Signals':>8} {'Avg ret':>10} "
          f"{'Win rate':>10} {'vs BH':>10}")
    print(f"  {'─'*50}")

    for ticker, signals_df in all_signals.items():
        buy   = signals_df[signals_df["signal"] == 1]
        col   = "return_63d"
        if col not in buy.columns:
            continue
        buy_r = buy[col].dropna()
        bh_r  = signals_df[col].dropna()
        if buy_r.empty:
            continue
        edge  = buy_r.mean() - bh_r.mean()
        print(f"  {ticker:<8} "
              f"{len(buy_r):>8} "
              f"{buy_r.mean():>+10.2%} "
              f"{(buy_r > 0).mean():>10.1%} "
              f"{edge:>+10.2%}")

    print("\nBacktest complete. Charts saved to charts/ folder.")

if __name__ == "__main__":
    run_full_backtest()