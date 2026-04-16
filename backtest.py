# backtest.py
# Full backtester + holding period analysis.
# Simulates trading based on XGBoost signals and measures:
#   - Cumulative P&L vs buy-and-hold
#   - Sharpe ratio (risk-adjusted return)
#   - Maximum drawdown (worst peak-to-trough loss)
#   - Win rate per holding period
#   - Quarterly breakdown

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_all_tickers
from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from earnings_loader import load_all_earnings
from features import add_features, get_feature_columns
from config import TICKERS, TRAIN_RATIO, VAL_RATIO, HOLDING_PERIODS

def get_signal_returns(ticker, df, macro_df=None,
                       sentiment=None, earnings=None):
    """
    Runs the saved model on the test set and computes forward
    returns at each holding period for every signal.
    """
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    model_path  = os.path.join("models", f"{ticker}_xgb.pkl")

    if not os.path.exists(model_path):
        print(f"  No model for {ticker}")
        return None

    sentiment_series = sentiment.get(ticker) if sentiment else None
    earnings_df      = earnings.get(ticker)  if earnings  else None

    df_feat = add_features(df.copy(), macro_df=macro_df,
                           sentiment_series=sentiment_series,
                           earnings_df=earnings_df)
    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment_series is not None,
        include_earnings=earnings_df is not None,
    )
    feat_cols = [c for c in feat_cols if c in df_feat.columns]

    scaler   = joblib.load(scaler_path)
    model    = joblib.load(model_path)
    X_scaled = scaler.transform(df_feat[feat_cols].values)
    probs    = model.predict_proba(X_scaled)[:, 1]

    df_feat["prob_up"] = probs
    df_feat["signal"]  = (probs >= 0.5).astype(int)

    n         = len(df_feat)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_df   = df_feat.iloc[val_end:].copy()
    close_all = df_feat["Close"]

    records = []
    for i, (date, row) in enumerate(test_df.iterrows()):
        abs_idx = val_end + i
        record  = {
            "date":    date,
            "close":   row["Close"],
            "signal":  int(row["signal"]),
            "prob_up": row["prob_up"],
            "quarter": f"{date.year}Q{(date.month-1)//3+1}",
        }
        for days in HOLDING_PERIODS:
            future_idx = abs_idx + days
            if future_idx < len(close_all):
                fwd = (close_all.iloc[future_idx] - row["Close"]) / row["Close"]
                record[f"return_{days}d"] = fwd
            else:
                record[f"return_{days}d"] = np.nan
        records.append(record)

    return pd.DataFrame(records)

def simulate_trades(signals_df, holding_days=21):
    """
    Simulates a simple strategy:
    - Buy when signal = 1, hold for holding_days, then exit
    - No overlapping positions (skip signal if already in trade)
    - Compare daily equity curve against buy-and-hold

    Returns a DataFrame with daily portfolio values for
    both the signal strategy and buy-and-hold.
    """
    col      = f"return_{holding_days}d"
    if col not in signals_df.columns:
        return None

    signals  = signals_df.copy().reset_index(drop=True)
    cash     = 1.0   # start with $1 (normalized)
    equity   = []
    in_trade_until = -1

    for i, row in signals.iterrows():
        if i <= in_trade_until:
            equity.append({"date": row["date"], "strategy": np.nan,
                           "bh": np.nan})
            continue

        if row["signal"] == 1 and not pd.isna(row[col]):
            ret      = row[col]
            cash     = cash * (1 + ret)
            in_trade_until = i + holding_days
            equity.append({"date": row["date"], "strategy": cash,
                           "bh": np.nan})
        else:
            equity.append({"date": row["date"], "strategy": cash,
                           "bh": np.nan})

    eq_df = pd.DataFrame(equity)

    # Buy and hold — invest $1 on first day, hold through entire test period
    first_close = signals_df["close"].iloc[0]
    eq_df["bh"] = signals_df["close"].values / first_close

    return eq_df

def sharpe_ratio(returns: pd.Series, periods_per_year=252) -> float:
    """
    Annualized Sharpe ratio assuming risk-free rate of 0.
    Higher is better. >1 is good, >2 is excellent.
    """
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in the equity curve.
    Expressed as a negative percentage.
    """
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def full_backtest_report(ticker, signals_df):
    """Prints complete backtest metrics for one ticker."""
    if signals_df is None or signals_df.empty:
        return

    buy   = signals_df[signals_df["signal"] == 1]
    total = len(signals_df)

    print(f"\n{'═'*60}")
    print(f"  BACKTEST REPORT — {ticker}")
    print(f"{'═'*60}")
    print(f"  Period: {signals_df['date'].min().date()} "
          f"to {signals_df['date'].max().date()}")
    print(f"  Total days:  {total}")
    print(f"  Buy signals: {len(buy)} ({len(buy)/total:.1%})")

    # Holding period performance table
    print(f"\n  {'Period':<8} {'Signals':>8} {'Avg ret':>10} "
          f"{'Win%':>8} {'BH avg':>10} {'Edge':>10} {'Sharpe':>8}")
    print(f"  {'─'*65}")

    for days in HOLDING_PERIODS:
        col = f"return_{days}d"
        if col not in signals_df.columns:
            continue

        buy_ret = buy[col].dropna()
        all_ret = signals_df[col].dropna()
        if buy_ret.empty:
            continue

        label = (f"{days}d"   if days < 21  else
                 f"{days//21}mo" if days < 63  else
                 f"{days//63}q"  if days < 126 else "6mo")

        sr    = sharpe_ratio(buy_ret)
        edge  = buy_ret.mean() - all_ret.mean()

        print(f"  {label:<8} {len(buy_ret):>8} "
              f"{buy_ret.mean():>+10.2%} "
              f"{(buy_ret>0).mean():>8.1%} "
              f"{all_ret.mean():>+10.2%} "
              f"{edge:>+10.2%} "
              f"{sr:>8.2f}")

    # Simulate the 1-month strategy
    eq = simulate_trades(signals_df, holding_days=21)
    if eq is not None:
        strat_final = eq["strategy"].dropna().iloc[-1]
        bh_final    = eq["bh"].iloc[-1]
        strat_ret   = eq["strategy"].dropna().pct_change().dropna()
        strat_mdd   = max_drawdown(eq["strategy"].dropna())

        print(f"\n  Simulated strategy (1mo hold, no overlapping positions):")
        print(f"  Strategy return:   {(strat_final-1):>+.2%}")
        print(f"  Buy-and-hold:      {(bh_final-1):>+.2%}")
        print(f"  Outperformance:    {(strat_final-bh_final):>+.2%}")
        print(f"  Sharpe ratio:      {sharpe_ratio(strat_ret):.2f}")
        print(f"  Max drawdown:      {strat_mdd:.2%}")

    # Quarterly breakdown
    long  = max(HOLDING_PERIODS)
    col   = f"return_{long}d"
    print(f"\n  Quarterly breakdown — {long}d hold:")
    print(f"  {'Quarter':<10} {'Signals':>8} {'Avg ret':>12} "
          f"{'Win%':>8} {'Best':>10} {'Worst':>10}")
    print(f"  {'─'*62}")

    for quarter, group in buy.groupby("quarter"):
        ret = group[col].dropna()
        if ret.empty:
            continue
        print(f"  {quarter:<10} {len(ret):>8} "
              f"{ret.mean():>+12.2%} "
              f"{(ret>0).mean():>8.1%} "
              f"{ret.max():>+10.2%} "
              f"{ret.min():>+10.2%}")

def plot_backtest(ticker, signals_df):
    """Generates the interactive 4-panel backtest chart."""
    if signals_df is None or signals_df.empty:
        return

    buy = signals_df[signals_df["signal"] == 1].copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cumulative return — signal strategy vs buy-and-hold (1mo hold)",
            "Avg return by quarter (1 quarter hold)",
            "Return distribution at each holding period",
            "Win rate by holding period",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Panel 1 — equity curves
    eq = simulate_trades(signals_df, holding_days=21)
    if eq is not None:
        strat = eq["strategy"].ffill()
        bh    = eq["bh"]
        fig.add_trace(go.Scatter(
            x=eq["date"], y=(strat - 1) * 100,
            mode="lines", name="Signal strategy",
            line=dict(color="#1D9E75", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=eq["date"], y=(bh - 1) * 100,
            mode="lines", name="Buy and hold",
            line=dict(color="#888780", width=1.5, dash="dash"),
        ), row=1, col=1)
        fig.add_hline(y=0, line_color="#E24B4A",
                      line_width=0.5, opacity=0.5, row=1, col=1)

    # Panel 2 — quarterly bar chart
    col_63 = "return_63d"
    if col_63 in buy.columns:
        qtr    = buy.groupby("quarter")[col_63].mean().dropna()
        colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in qtr.values]
        fig.add_trace(go.Bar(
            x=qtr.index, y=qtr.values * 100,
            marker_color=colors,
            name="Avg quarterly return",
            showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=0, line_color="#888780",
                      line_width=0.5, row=1, col=2)

    # Panel 3 — violin distributions
    colors_dist = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"]
    for days, color in zip(HOLDING_PERIODS, colors_dist):
        col = f"return_{days}d"
        if col not in buy.columns:
            continue
        rets  = buy[col].dropna() * 100
        label = (f"{days}d"   if days < 21  else
                 f"{days//21}mo" if days < 63  else
                 f"{days//63}q"  if days < 126 else "6mo")
        fig.add_trace(go.Violin(
            y=rets, name=label,
            box_visible=True, meanline_visible=True,
            line_color=color, fillcolor=color, opacity=0.4,
        ), row=2, col=1)

    # Panel 4 — win rates
    win_rates, labels = [], []
    for days in HOLDING_PERIODS:
        col = f"return_{days}d"
        if col not in buy.columns:
            continue
        wr    = (buy[col].dropna() > 0).mean() * 100
        label = (f"{days}d"   if days < 21  else
                 f"{days//21}mo" if days < 63  else
                 f"{days//63}q"  if days < 126 else "6mo")
        win_rates.append(wr)
        labels.append(label)

    fig.add_trace(go.Bar(
        x=labels, y=win_rates,
        marker_color=["#1D9E75" if w >= 50 else "#E24B4A" for w in win_rates],
        text=[f"{w:.1f}%" for w in win_rates],
        textposition="outside",
        showlegend=False,
    ), row=2, col=2)
    fig.add_hline(y=50, line_dash="dash",
                  line_color="#888780", line_width=1,
                  opacity=0.5, row=2, col=2)

    fig.update_layout(
        height=800,
        title=f"{ticker} — Full Backtest (test set only)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="Return %",        row=1, col=1)
    fig.update_yaxes(title_text="Avg return %",    row=1, col=2)
    fig.update_yaxes(title_text="Return %",        row=2, col=1)
    fig.update_yaxes(title_text="Win rate %",
                     range=[0, 110],               row=2, col=2)

    os.makedirs("charts", exist_ok=True)
    out = os.path.join("charts", f"{ticker}_backtest.html")
    fig.write_html(out)
    print(f"  Chart saved: {out}")

def run_full_backtest():
    print("Loading data...")
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()
    earnings  = load_all_earnings(all_data)

    all_signals = {}

    for ticker, df in all_data.items():
        print(f"\nAnalyzing {ticker}...")
        signals_df = get_signal_returns(
            ticker, df,
            macro_df=macro_df,
            sentiment=sentiment,
            earnings=earnings,
        )
        if signals_df is not None:
            all_signals[ticker] = signals_df
            full_backtest_report(ticker, signals_df)
            plot_backtest(ticker, signals_df)

    # Cross-ticker summary
    print(f"\n{'═'*60}")
    print("  CROSS-TICKER SUMMARY — 1 quarter hold")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<8} {'Signals':>8} {'Avg ret':>10} "
          f"{'Win%':>8} {'Edge':>10} {'Sharpe':>8}")
    print(f"  {'─'*55}")

    for ticker, sdf in all_signals.items():
        buy = sdf[sdf["signal"] == 1]
        col = "return_63d"
        if col not in buy.columns:
            continue
        buy_r = buy[col].dropna()
        bh_r  = sdf[col].dropna()
        if buy_r.empty:
            continue
        print(f"  {ticker:<8} {len(buy_r):>8} "
              f"{buy_r.mean():>+10.2%} "
              f"{(buy_r>0).mean():>8.1%} "
              f"{buy_r.mean()-bh_r.mean():>+10.2%} "
              f"{sharpe_ratio(buy_r):>8.2f}")

    print("\nBacktest complete. Charts saved to charts/ folder.")

if __name__ == "__main__":
    run_full_backtest()