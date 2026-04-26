# app/services/chart.py
#
# Generates interactive Plotly candlestick charts with prediction signals
# overlaid and a probability time series in a subplot below.
#
# Charts are colour-coded by train/val/test region so you can visually
# confirm that signals in the shaded test region are truly out-of-sample.
#
# Output: charts/<TICKER>_predictions.html

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import TICKERS, TRAIN_RATIO, VAL_RATIO
from app.data import load_all_tickers, fetch_macro, load_all_sentiment
from app.ml.features import add_features, get_feature_columns


def get_predictions(
    ticker: str,
    df: pd.DataFrame,
    macro_df=None,
    sentiment=None,
) -> pd.DataFrame | None:
    """
    Load the trained XGBoost model for ticker and generate predictions
    over the full historical dataset.

    Returns the feature DataFrame with added prob_up, signal, and section
    columns, or None if no model file exists.
    """
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    model_path  = os.path.join("models", f"{ticker}_xgb.pkl")
    if not os.path.exists(model_path):
        print(f"  No model found for {ticker}")
        return None

    sentiment_series = sentiment.get(ticker) if sentiment else None
    df_feat = add_features(
        df.copy(),
        macro_df=macro_df,
        sentiment_series=sentiment_series,
    )
    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment_series is not None,
    )
    feat_cols = [c for c in feat_cols if c in df_feat.columns]

    scaler   = joblib.load(scaler_path)
    model    = joblib.load(model_path)
    X_scaled = scaler.transform(df_feat[feat_cols].values)
    probs    = model.predict_proba(X_scaled)[:, 1]

    df_feat["prob_up"] = probs
    df_feat["signal"]  = (probs >= 0.5).astype(int)

    n         = len(df_feat)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    df_feat["section"] = "train"
    df_feat.iloc[train_end:val_end, df_feat.columns.get_loc("section")] = "val"
    df_feat.iloc[val_end:,          df_feat.columns.get_loc("section")] = "test"

    return df_feat


def build_chart(ticker: str, df_feat: pd.DataFrame) -> go.Figure:
    """
    Build a three-panel Plotly figure:
      Row 1 — Candlestick with buy/sell signal markers and region shading
      Row 2 — Volume bars coloured by up/down day
      Row 3 — P(price goes up) time series with 0.5 threshold line
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25],
        subplot_titles=[
            f"{ticker} — XGBoost prediction signals",
            "Volume",
            "P(price goes up tomorrow)",
        ],
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_feat.index,
        open=df_feat["Open"],
        high=df_feat["High"],
        low=df_feat["Low"],
        close=df_feat["Close"],
        name="Price",
        increasing_line_color="#1D9E75",
        decreasing_line_color="#E24B4A",
        showlegend=False,
    ), row=1, col=1)

    # ── Region shading ────────────────────────────────────────────────────
    sections = {
        "train": ("rgba(200,200,200,0.1)", "Train"),
        "val":   ("rgba(100,150,255,0.1)", "Validation"),
        "test":  ("rgba(255,200,100,0.1)", "Test"),
    }
    for section, (color, label) in sections.items():
        mask = df_feat["section"] == section
        if mask.any():
            fig.add_vrect(
                x0=df_feat.index[mask][0],
                x1=df_feat.index[mask][-1],
                fillcolor=color,
                line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=10,
            )

    # ── Signal markers (test set only) ────────────────────────────────────
    test_mask = df_feat["section"] == "test"
    buy_mask  = test_mask & (df_feat["signal"] == 1)
    sell_mask = test_mask & (df_feat["signal"] == 0)

    fig.add_trace(go.Scatter(
        x=df_feat.index[buy_mask],
        y=df_feat["Low"][buy_mask] * 0.98,
        mode="markers",
        marker=dict(symbol="triangle-up", size=8, color="#1D9E75"),
        name="Buy signal",
        hovertemplate="Buy<br>%{x}<br>P(up)=%{customdata:.3f}",
        customdata=df_feat["prob_up"][buy_mask],
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_feat.index[sell_mask],
        y=df_feat["High"][sell_mask] * 1.02,
        mode="markers",
        marker=dict(symbol="triangle-down", size=8, color="#E24B4A"),
        name="Sell signal",
        hovertemplate="Sell<br>%{x}<br>P(up)=%{customdata:.3f}",
        customdata=df_feat["prob_up"][sell_mask],
    ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────
    colors = [
        "#1D9E75" if c >= o else "#E24B4A"
        for c, o in zip(df_feat["Close"], df_feat["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df_feat.index,
        y=df_feat["Volume"],
        marker_color=colors,
        name="Volume",
        showlegend=False,
        opacity=0.6,
    ), row=2, col=1)

    # ── Probability time series ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_feat.index,
        y=df_feat["prob_up"],
        mode="lines",
        line=dict(color="#378ADD", width=1),
        name="P(up)",
    ), row=3, col=1)
    fig.add_hline(
        y=0.5, line_dash="dash",
        line_color="#E24B4A", line_width=1, opacity=0.5,
        row=3, col=1,
    )

    fig.update_layout(
        height=800,
        title=f"{ticker} — XGBoost next-day direction predictions",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",    row=2, col=1)
    fig.update_yaxes(title_text="P(up)", range=[0, 1], row=3, col=1)

    return fig


def generate_all_charts() -> None:
    """Generate and save charts for every ticker in the watchlist."""
    print("Loading data...")
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()

    os.makedirs("charts", exist_ok=True)

    for ticker, df in all_data.items():
        print(f"  Building chart for {ticker}...")
        df_feat = get_predictions(ticker, df, macro_df=macro_df, sentiment=sentiment)
        if df_feat is None:
            continue
        fig      = build_chart(ticker, df_feat)
        out_path = os.path.join("charts", f"{ticker}_predictions.html")
        fig.write_html(out_path)
        print(f"  Saved: {out_path}")

    print("\nAll charts saved to charts/ folder.")


if __name__ == "__main__":
    generate_all_charts()
