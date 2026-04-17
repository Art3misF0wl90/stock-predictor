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

TICKERS_NO_EARNINGS = ["TSLA", "SPY"]

MIN_PROB = 0.55
MAX_VIX  = 30

TICKER_MIN_PROB = {
    "AAPL":  0.55,
    "MSFT":  0.55,
    "TSLA":  0.55,
    "JPM":   0.55,
    "NVDA":  0.50,
    "GOOGL": 0.50,
    "AMZN":  0.52,
    "META":  0.52,
    "SPY":   0.55,
    "AMD":   0.55,
}

def get_signal_returns(ticker, df, macro_df=None,
                       sentiment=None, earnings=None,
                       max_vix=MAX_VIX,
                       use_trend_filter=True,
                       use_rsi_filter=True,
                       use_earnings_filter=True):
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    model_path  = os.path.join("models", f"{ticker}_model.pkl")
    config_path = os.path.join("models", f"{ticker}_config.pkl")

    if not os.path.exists(model_path):
        print(f"  No model for {ticker}")
        return None

    scaler    = joblib.load(scaler_path)
    model     = joblib.load(model_path)
    cfg       = joblib.load(config_path)
    feat_cols = cfg["feat_cols"]
    fwd_days  = cfg["fwd_days"]

    sentiment_series = sentiment.get(ticker) if sentiment else None

    # Use earnings only if the saved model was trained with them
    has_earnings = any(
        "eps" in c or "pead" in c or "earnings" in c
        for c in feat_cols
    )
    earnings_df = (
        earnings.get(ticker) if (earnings and has_earnings) else None
    )

    df_feat = add_features(df.copy(), macro_df=macro_df,
                           sentiment_series=sentiment_series,
                           earnings_df=earnings_df,
                           forward_days=fwd_days)

    feat_cols = [c for c in feat_cols if c in df_feat.columns]
    X_scaled  = scaler.transform(df_feat[feat_cols].values)
    probs     = model.predict_proba(X_scaled)[:, 1]

    df_feat["prob_up"] = probs
    df_feat["signal"]  = (probs >= 0.5).astype(int)

    n         = len(df_feat)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_df   = df_feat.iloc[val_end:].copy()
    close_all = df_feat["Close"]

    ticker_min_prob    = TICKER_MIN_PROB.get(ticker, MIN_PROB)
    records            = []
    consecutive_losses = 0

    for i, (date_idx, row) in enumerate(test_df.iterrows()):
        abs_idx    = val_end + i
        raw_signal = int(row["signal"])
        prob_up    = float(row["prob_up"])
        filtered   = False
        filter_reason = None

        if raw_signal == 1:
            # Confidence filter — per ticker
            if prob_up < ticker_min_prob:
                filtered      = True
                filter_reason = f"low_confidence ({prob_up:.2f} < {ticker_min_prob})"

            # VIX crisis filter
            elif "vix" in row.index and not pd.isna(row.get("vix", np.nan)):
                if float(row["vix"]) > max_vix:
                    filtered      = True
                    filter_reason = f"high_vix ({row['vix']:.1f} > {max_vix})"

            # Trend filter — price below 50-day MA
            elif use_trend_filter and "price_to_ma50" in row.index:
                if float(row.get("price_to_ma50", 1.0)) < 1.0:
                    filtered      = True
                    filter_reason = "below_50ma (downtrend)"

            # RSI overbought filter
            elif use_rsi_filter and "rsi_14" in row.index:
                if float(row.get("rsi_14", 50)) > 70:
                    filtered      = True
                    filter_reason = f"overbought_rsi ({row['rsi_14']:.1f})"

            # Earnings blackout filter
            elif use_earnings_filter and "days_to_earnings" in row.index:
                dte = float(row.get("days_to_earnings", 90))
                if 0 < dte <= 3:
                    filtered      = True
                    filter_reason = f"earnings_blackout ({dte:.0f}d)"

            # Consecutive losses pause
            elif consecutive_losses >= 3:
                filtered      = True
                filter_reason = f"consecutive_losses ({consecutive_losses})"

        final_signal = 1 if (raw_signal == 1 and not filtered) else 0

        record = {
            "date":          date_idx,
            "close":         row["Close"],
            "raw_signal":    raw_signal,
            "signal":        final_signal,
            "prob_up":       prob_up,
            "filtered":      filtered,
            "filter_reason": filter_reason,
            "quarter":       f"{date_idx.year}Q{(date_idx.month-1)//3+1}",
        }

        for days in HOLDING_PERIODS:
            future_idx = abs_idx + days
            if future_idx < len(close_all):
                fwd = (close_all.iloc[future_idx] - row["Close"]) / row["Close"]
                record[f"return_{days}d"] = fwd
            else:
                record[f"return_{days}d"] = np.nan

        col_21 = "return_21d"
        if final_signal == 1 and col_21 in record:
            ret = record[col_21]
            if not np.isnan(ret):
                consecutive_losses = consecutive_losses + 1 if ret < 0 else 0

        records.append(record)

    result_df   = pd.DataFrame(records)
    total_raw   = int(result_df["raw_signal"].sum())
    total_final = int(result_df["signal"].sum())
    filtered_df = result_df[result_df["filtered"] == True]

    if total_raw > 0:
        print(f"  Signal filter summary for {ticker}:")
        print(f"    Raw buy signals:  {total_raw}")
        print(f"    Filtered out:     {len(filtered_df)}")
        print(f"    Accepted:         {total_final}")
        if len(filtered_df) > 0:
            for reason, count in filtered_df["filter_reason"].value_counts().items():
                print(f"      - {reason}: {count}")

    return result_df

def simulate_trades(signals_df, holding_days=21):
    col = f"return_{holding_days}d"
    if col not in signals_df.columns:
        return None

    signals        = signals_df.copy().reset_index(drop=True)
    cash           = 1.0
    equity         = []
    in_trade_until = -1

    for i, row in signals.iterrows():
        if i <= in_trade_until:
            equity.append({"date": row["date"], "strategy": np.nan,
                           "bh": np.nan})
            continue
        if row["signal"] == 1 and not pd.isna(row[col]):
            cash           = cash * (1 + row[col])
            in_trade_until = i + holding_days
            equity.append({"date": row["date"], "strategy": cash,
                           "bh": np.nan})
        else:
            equity.append({"date": row["date"], "strategy": cash,
                           "bh": np.nan})

    eq_df       = pd.DataFrame(equity)
    first_close = signals_df["close"].iloc[0]
    eq_df["bh"] = signals_df["close"].values / first_close
    return eq_df

def sharpe_ratio(returns: pd.Series, periods_per_year=252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def full_backtest_report(ticker, signals_df):
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

    if "filtered" in signals_df.columns:
        raw_buys = int(signals_df["raw_signal"].sum())
        filtered = raw_buys - len(buy)
        if raw_buys > 0:
            print(f"  Raw signals: {raw_buys} | "
                  f"Filtered: {filtered} "
                  f"({filtered/raw_buys*100:.1f}% rejected)")

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
        label = (f"{days}d"      if days < 21  else
                 f"{days//21}mo" if days < 63  else
                 f"{days//63}q"  if days < 126 else "6mo")
        sr   = sharpe_ratio(buy_ret)
        edge = buy_ret.mean() - all_ret.mean()
        print(f"  {label:<8} {len(buy_ret):>8} "
              f"{buy_ret.mean():>+10.2%} "
              f"{(buy_ret>0).mean():>8.1%} "
              f"{all_ret.mean():>+10.2%} "
              f"{edge:>+10.2%} "
              f"{sr:>8.2f}")

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

    long = max(HOLDING_PERIODS)
    col  = f"return_{long}d"
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

    col_63 = "return_63d"
    if col_63 in buy.columns and not buy.empty:
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

    colors_dist = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"]
    for days, color in zip(HOLDING_PERIODS, colors_dist):
        col = f"return_{days}d"
        if col not in buy.columns or buy.empty:
            continue
        rets  = buy[col].dropna() * 100
        if rets.empty:
            continue
        label = (f"{days}d"      if days < 21  else
                 f"{days//21}mo" if days < 63  else
                 f"{days//63}q"  if days < 126 else "6mo")
        fig.add_trace(go.Violin(
            y=rets, name=label,
            box_visible=True, meanline_visible=True,
            line_color=color, fillcolor=color, opacity=0.4,
        ), row=2, col=1)

    win_rates, labels = [], []
    for days in HOLDING_PERIODS:
        col = f"return_{days}d"
        if col not in buy.columns or buy.empty:
            continue
        ret = buy[col].dropna()
        if ret.empty:
            continue
        wr    = (ret > 0).mean() * 100
        label = (f"{days}d"      if days < 21  else
                 f"{days//21}mo" if days < 63  else
                 f"{days//63}q"  if days < 126 else "6mo")
        win_rates.append(wr)
        labels.append(label)

    if win_rates:
        fig.add_trace(go.Bar(
            x=labels, y=win_rates,
            marker_color=["#1D9E75" if w >= 50 else "#E24B4A"
                          for w in win_rates],
            text=[f"{w:.1f}%" for w in win_rates],
            textposition="outside",
            showlegend=False,
        ), row=2, col=2)
        fig.add_hline(y=50, line_dash="dash",
                      line_color="#888780", line_width=1,
                      opacity=0.5, row=2, col=2)

    fig.update_layout(
        height=800,
        title=f"{ticker} — Filtered Backtest (test set only)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="Return %",     row=1, col=1)
    fig.update_yaxes(title_text="Avg return %", row=1, col=2)
    fig.update_yaxes(title_text="Return %",     row=2, col=1)
    fig.update_yaxes(title_text="Win rate %",
                     range=[0, 110],            row=2, col=2)

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

    print(f"\n{'═'*60}")
    print("  CROSS-TICKER SUMMARY — 1 quarter hold (filtered signals)")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<8} {'Signals':>8} {'Avg ret':>10} "
          f"{'Win%':>8} {'Edge':>10} {'Sharpe':>8}")
    print(f"  {'─'*55}")

    for ticker, sdf in all_signals.items():
        buy   = sdf[sdf["signal"] == 1]
        col   = "return_63d"
        if col not in buy.columns or buy.empty:
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