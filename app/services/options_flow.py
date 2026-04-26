# app/services/options_flow.py
#
# Options chain scanner and dashboard generator.
#
# Computes per-ticker:
#   PCR (Put/Call Ratio)    — market sentiment proxy
#   Unusual volume          — contracts with vol >> open interest
#   GEX (Gamma Exposure)    — whether market makers will dampen or amplify moves
#   IV Rank                 — where current IV sits in its recent range
#   Expected Move           — market-implied ±1σ price range by nearest expiry
#
# Combines the above with the ML model signal for a combined prediction.
#
# Public API:
#   scan_ticker(ticker)   — full scan for one ticker, returns result dict
#   scan_all(tickers)     — scan all tickers, returns {ticker: result} dict
#   generate_options_dashboard(...)  — build and save Plotly dashboard HTML
#   open_dashboard(...)   — generate + open in browser

import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import joblib
from plotly.subplots import make_subplots

from config import TICKERS

# ---------------------------------------------------------------------------
# Chart colour palette (dark theme matching the web dashboard)
# ---------------------------------------------------------------------------

_BG      = "#0f1117"
_PANEL   = "#1a1d2e"
_GRID    = "#2d3748"
_TEXT    = "#e2e8f0"
_SUBTEXT = "#718096"
_GREEN   = "#1D9E75"
_RED     = "#E24B4A"
_BLUE    = "#378ADD"
_ORANGE  = "#EF9F27"
_PURPLE  = "#9F7AEA"


def _iv_color(val, low_good=True):
    """Return colour based on whether a low or high IV rank is desirable."""
    if val is None:
        return _SUBTEXT
    if low_good:
        return _GREEN if val < 30 else _RED if val > 70 else _ORANGE
    return _GREEN if val > 70 else _RED if val < 30 else _ORANGE


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _fetch_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        return chain.calls, chain.puts
    except Exception as e:
        print(f"  Could not fetch chain for {ticker} {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def _get_spot_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).info.get("regularMarketPrice", 0))
    except Exception:
        return 0.0


def _get_nearest_expiries(ticker: str, n: int = 4) -> list[str]:
    try:
        return list(yf.Ticker(ticker).options[:n])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_pcr(calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """
    Put/Call Ratio by volume and open interest.

    PCR < 0.7  = bullish (more calls being bought)
    PCR > 1.2  = bearish / heavy hedging
    """
    call_vol = calls["volume"].fillna(0).sum()
    put_vol  = puts["volume"].fillna(0).sum()
    call_oi  = calls["openInterest"].fillna(0).sum()
    put_oi   = puts["openInterest"].fillna(0).sum()

    pcr_vol = put_vol / (call_vol + 1e-9)
    pcr_oi  = put_oi  / (call_oi  + 1e-9)

    sentiment = "Bullish" if pcr_vol < 0.7 else "Bearish" if pcr_vol > 1.2 else "Neutral"

    return {
        "pcr_volume":  round(pcr_vol, 3),
        "pcr_oi":      round(pcr_oi, 3),
        "call_volume": int(call_vol),
        "put_volume":  int(put_vol),
        "call_oi":     int(call_oi),
        "put_oi":      int(put_oi),
        "sentiment":   sentiment,
    }


def find_unusual_volume(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Flag contracts where volume > threshold × open interest.

    Filters out zero-OI contracts to avoid division artifacts.
    Returns a DataFrame sorted by vol/OI ratio descending.
    """
    records = []
    for df, kind in [(calls, "CALL"), (puts, "PUT")]:
        if df.empty:
            continue
        df = df.copy()
        df["volume"]       = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)
        df = df[df["openInterest"] > 0]
        if df.empty:
            continue
        df["vol_oi_ratio"] = df["volume"] / (df["openInterest"] + 1e-9)
        for _, row in df[df["vol_oi_ratio"] >= threshold].iterrows():
            records.append({
                "type":          kind,
                "strike":        row["strike"],
                "volume":        int(row["volume"]),
                "open_interest": int(row["openInterest"]),
                "vol_oi_ratio":  round(row["vol_oi_ratio"], 2),
                "iv":            round(row["impliedVolatility"], 4),
                "last_price":    row["lastPrice"],
                "in_the_money":  row["inTheMoney"],
            })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("vol_oi_ratio", ascending=False).reset_index(drop=True)


def compute_gex(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
) -> dict:
    """
    Gamma Exposure — estimates net gamma position of market makers.

    Positive GEX → market makers long gamma → price dampening (pinning)
    Negative GEX → market makers short gamma → price amplifying (trending)

    Filters to strikes within ±15% of spot to avoid deep OTM noise.
    """
    empty = {"total_gex": 0, "call_wall": None, "put_wall": None, "gex_bias": "Unknown", "spot": spot}
    if calls.empty and puts.empty:
        return empty

    def _filter(df):
        return df[
            (df["strike"] >= spot * 0.85) &
            (df["strike"] <= spot * 1.15) &
            (df["openInterest"] > 0) &
            (df["impliedVolatility"] > 0.01)
        ].copy()

    call_gex = _filter(calls)
    put_gex  = _filter(puts)

    if call_gex.empty and put_gex.empty:
        return empty

    call_gex["gex"] = call_gex["openInterest"] * call_gex["impliedVolatility"] * (spot ** 2) * 0.01
    put_gex["gex"]  = -(put_gex["openInterest"] * put_gex["impliedVolatility"] * (spot ** 2) * 0.01)

    total_gex       = call_gex["gex"].sum() + put_gex["gex"].sum()
    max_call_strike = float(call_gex.loc[call_gex["gex"].idxmax(), "strike"]) if not call_gex.empty else None
    max_put_strike  = float(put_gex.loc[put_gex["gex"].idxmin(), "strike"])  if not put_gex.empty  else None

    return {
        "total_gex": round(float(total_gex), 2),   # BUG FIX: float() takes 1 arg
        "call_wall": max_call_strike,
        "put_wall":  max_put_strike,
        "gex_bias":  "Pinning" if total_gex > 0 else "Trending",
        "spot":      spot,
    }


def compute_iv_rank(ticker: str) -> dict:
    """
    IV Rank — where current implied volatility sits in its recent range.

    Skips the first (same-day) expiry; uses the next 3.
    Filters to near-the-money strikes only.

    IVR < 30 = cheap options; IVR > 70 = expensive options
    """
    try:
        stock    = yf.Ticker(ticker)
        expiries = stock.options
        if not expiries:
            return {"iv_rank": None, "current_iv": None, "iv_label": "Unknown"}

        spot    = _get_spot_price(ticker)
        iv_list = []

        for exp in expiries[1:4]:
            calls, _ = _fetch_chain(ticker, exp)
            if calls.empty:
                continue
            calls = calls[
                (calls["strike"] >= spot * 0.95) &
                (calls["strike"] <= spot * 1.05) &
                (calls["impliedVolatility"] > 0.01)
            ].copy()
            if calls.empty:
                continue
            calls["dist"] = (calls["strike"] - spot).abs()
            atm = calls.loc[calls["dist"].idxmin()]
            iv  = float(atm["impliedVolatility"])
            if iv > 0.01:
                iv_list.append(iv)

        if not iv_list:
            return {"iv_rank": None, "current_iv": None, "iv_label": "Unknown"}

        current_iv = np.mean(iv_list)
        iv_rank    = (current_iv - min(iv_list)) / (max(iv_list) - min(iv_list) + 1e-9) * 100
        iv_label   = (
            "Low (cheap options)"    if iv_rank < 30 else
            "High (expensive options)" if iv_rank > 70 else
            "Normal"
        )

        return {
            "iv_rank":    round(iv_rank, 1),
            "current_iv": round(current_iv, 4),
            "iv_label":   iv_label,
        }
    except Exception as e:
        return {"iv_rank": None, "current_iv": None, "iv_label": f"Error: {e}"}


def compute_expected_move(spot: float, iv: float, days_to_expiry: int) -> dict:
    """
    Market-implied ±1σ price range by expiry.

    Formula: EM = spot × IV × sqrt(days / 365)
    There is ~68% probability price stays within this range.
    """
    if not iv or not spot or iv < 0.01:
        return {}
    em     = spot * iv * np.sqrt(days_to_expiry / 365)
    em_pct = (em / spot) * 100
    return {
        "expected_move":     round(em, 2),
        "expected_move_pct": round(em_pct, 2),
        "upper_target":      round(spot + em, 2),
        "lower_target":      round(spot - em, 2),
        "days_to_expiry":    days_to_expiry,
    }


# ---------------------------------------------------------------------------
# ML model signal
# ---------------------------------------------------------------------------

def get_model_signal(ticker: str) -> dict:
    """
    Load and run the trained model for ticker on the latest bar.

    Returns prob_up, direction (UP/DOWN), model name, and horizon.
    """
    ticker      = ticker.upper()
    model_path  = os.path.join("models", f"{ticker}_model.pkl")
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    config_path = os.path.join("models", f"{ticker}_config.pkl")

    if not os.path.exists(model_path):
        return {"error": f"No model found for {ticker}"}

    try:
        from app.data import fetch_macro, load_all_sentiment, build_earnings_features
        from app.ml.features import add_features
        from app.ml import fetch_latest_data

        model      = joblib.load(model_path)
        scaler     = joblib.load(scaler_path)
        cfg        = joblib.load(config_path)
        feat_cols  = cfg["feat_cols"]
        fwd_days   = cfg["fwd_days"]
        model_name = cfg["model_name"]

        df        = fetch_latest_data(ticker)
        macro_df  = fetch_macro()
        sentiment = load_all_sentiment()
        sent      = sentiment.get(ticker)

        has_earnings = any("eps" in c or "pead" in c or "earnings" in c for c in feat_cols)
        earn = build_earnings_features(ticker, df) if has_earnings else None

        df_feat = add_features(
            df, macro_df=macro_df, sentiment_series=sent,
            earnings_df=earn, forward_days=fwd_days, predict_mode=True,
        )
        if df_feat.empty:
            return {"error": "Could not generate features"}

        fc      = [c for c in feat_cols if c in df_feat.columns]
        prob_up = float(model.predict_proba(scaler.transform(df_feat[fc].iloc[[-1]].values))[0][1])
        horizon_map = {1: "1d", 21: "21d", 63: "63d", 126: "126d"}

        return {
            "prob_up":   round(prob_up, 4),
            "direction": "UP" if prob_up >= 0.5 else "DOWN",
            "model":     model_name,
            "horizon":   horizon_map.get(fwd_days, f"{fwd_days}d"),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Combined prediction
# ---------------------------------------------------------------------------

def generate_prediction(
    model_signal: dict,
    flow_bias: str,
    expected_move: dict,
    spot: float,
) -> dict:
    """
    Combine ML model signal and options flow bias into one prediction.

    Agreement between model and flow = higher conviction.
    Disagreement = conflicted signal, exercise caution.
    """
    if "error" in model_signal or not expected_move:
        return {"status": "insufficient_data"}

    prob_up          = model_signal["prob_up"]
    model_dir        = model_signal["direction"]
    model_is_bullish = model_dir == "UP"
    flow_is_bullish  = flow_bias == "Bullish"
    agreement        = flow_is_bullish == model_is_bullish

    price_target = (
        expected_move.get("upper_target") if model_is_bullish
        else expected_move.get("lower_target")
    )
    conviction = (
        ("HIGH" if abs(prob_up - 0.5) > 0.1 else "MODERATE")
        if agreement else "LOW — model and flow disagree"
    )

    return {
        "model_direction": model_dir,
        "model_prob_up":   prob_up,
        "model_horizon":   model_signal["horizon"],
        "flow_direction":  "UP" if flow_is_bullish else "DOWN",
        "flow_bias":       flow_bias,
        "agreement":       agreement,
        "conviction":      conviction,
        "price_target":    price_target,
        "target_label":    "UP target" if model_is_bullish else "DOWN target",
        "move_pct":        expected_move.get("expected_move_pct", 0),
        "dte":             expected_move.get("days_to_expiry", 0),
        "spot":            spot,
    }


# ---------------------------------------------------------------------------
# Full scanner
# ---------------------------------------------------------------------------

def scan_ticker(ticker: str) -> dict:
    """Run the full options flow scan for one ticker."""
    ticker   = ticker.upper()
    print(f"  Scanning {ticker}...")

    expiries = _get_nearest_expiries(ticker, n=4)
    if not expiries:
        return {"ticker": ticker, "error": "No options available"}

    spot        = _get_spot_price(ticker)
    nearest     = expiries[1] if len(expiries) > 1 else expiries[0]
    calls, puts = _fetch_chain(ticker, nearest)

    exp_date = datetime.strptime(nearest, "%Y-%m-%d").date()
    dte      = max((exp_date - date.today()).days, 1)

    pcr          = compute_pcr(calls, puts)
    unusual      = find_unusual_volume(calls, puts, threshold=2.0)
    gex          = compute_gex(calls, puts, spot)
    iv_rank      = compute_iv_rank(ticker)
    em           = compute_expected_move(spot, iv_rank.get("current_iv", 0), dte)
    model_signal = get_model_signal(ticker)

    # Simple flow score: +1 for bullish PCR, −1 for bearish, adjust for unusual volume
    flow_score = 0
    if pcr["pcr_volume"] < 0.7:
        flow_score += 1
    elif pcr["pcr_volume"] > 1.2:
        flow_score -= 1
    if not unusual.empty:
        flow_score += len(unusual[unusual["type"] == "CALL"]) * 0.5
        flow_score -= len(unusual[unusual["type"] == "PUT"])  * 0.5

    flow_bias  = "Bullish" if flow_score > 0 else "Bearish" if flow_score < 0 else "Neutral"
    prediction = generate_prediction(model_signal, flow_bias, em, spot)

    return {
        "ticker":         ticker,
        "date":           str(date.today()),
        "spot":           spot,
        "nearest_expiry": nearest,
        "dte":            dte,
        "pcr":            pcr,
        "unusual_volume": unusual.to_dict("records") if not unusual.empty else [],
        "gex":            gex,
        "iv_rank":        iv_rank,
        "expected_move":  em,
        "flow_score":     round(flow_score, 2),
        "flow_bias":      flow_bias,
        "model_signal":   model_signal,
        "prediction":     prediction,
    }


def scan_all(tickers: list | None = None) -> dict:
    """Scan every ticker in the list (defaults to full watchlist)."""
    if tickers is None:
        tickers = TICKERS
    return {ticker: scan_ticker(ticker) for ticker in tickers}


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_flow_report(results: dict) -> None:
    print(f"\n{'═'*70}")
    print(f"  OPTIONS FLOW + MODEL PREDICTION REPORT — {date.today()}")
    print(f"{'═'*70}")

    for ticker, data in results.items():
        if "error" in data:
            print(f"  {ticker}: ERROR — {data['error']}")
            continue

        pcr  = data["pcr"]
        gex  = data["gex"]
        ivr  = data["iv_rank"]
        em   = data.get("expected_move", {})
        ms   = data.get("model_signal", {})
        pred = data.get("prediction", {})
        spot = data["spot"]

        print(f"\n  {'─'*68}")
        print(f"  {ticker} — ${spot}  |  Expiry: {data['nearest_expiry']} ({data['dte']}d)")
        print(f"  {'─'*68}")

        if "error" not in ms:
            arrow = "▲" if ms["direction"] == "UP" else "▼"
            print(
                f"  MODEL  {arrow} {ms['direction']:<6} "
                f"prob_up={ms['prob_up']:.1%}  "
                f"horizon={ms['horizon']}  ({ms['model']})"
            )
        else:
            print(f"  MODEL  N/A — {ms.get('error', 'unknown')}")

        flow_arrow = "▲" if data["flow_bias"] == "Bullish" else ("▼" if data["flow_bias"] == "Bearish" else "─")
        print(
            f"  FLOW   {flow_arrow} {data['flow_bias']:<6} "
            f"PCR={pcr['pcr_volume']:.3f}  "
            f"IVR={ivr.get('iv_rank', 'N/A')}  {ivr.get('iv_label', '')}"
        )
        print(
            f"  GEX    {gex.get('gex_bias', 'N/A'):<20} "
            f"Call wall=${gex.get('call_wall', 'N/A')}  "
            f"Put wall=${gex.get('put_wall', 'N/A')}"
        )
        if em:
            print(
                f"  MOVE   ±${em['expected_move']} ({em['expected_move_pct']}%)  "
                f"→  DOWN: ${em['lower_target']}  |  UP: ${em['upper_target']}"
            )
        if pred.get("status") != "insufficient_data":
            agree_str = "✓ AGREE" if pred["agreement"] else "✗ DISAGREE"
            print(f"  {'─'*68}")
            print(f"  PREDICTION  {agree_str}  |  Conviction: {pred['conviction']}")
            print(f"    Model says {pred['model_direction']} ({pred['model_prob_up']:.1%}) over {pred['model_horizon']}")
            print(f"    Flow  says {pred['flow_direction']} (PCR {pcr['pcr_volume']:.3f})")
            if pred["price_target"]:
                print(
                    f"    {pred['target_label']}: ${pred['price_target']} "
                    f"({pred['move_pct']}% move in {pred['dte']}d)"
                )
        if data["unusual_volume"]:
            print(f"  UNUSUAL  {len(data['unusual_volume'])} contracts flagged:")
            for u in data["unusual_volume"][:3]:
                print(
                    f"    {u['type']} ${u['strike']} — "
                    f"vol={u['volume']:,} vs OI={u['open_interest']:,} "
                    f"({u['vol_oi_ratio']}x)  IV={u['iv']:.3f}"
                )

    print(f"\n{'═'*70}\n")


# ---------------------------------------------------------------------------
# Plotly dashboard
# ---------------------------------------------------------------------------

def build_dashboard(results: dict) -> go.Figure:
    """Build the 5-row options flow dashboard figure."""
    tickers = [t for t in results if "error" not in results[t]]
    n       = len(tickers)
    if n == 0:
        print("No valid ticker data to chart.")
        return go.Figure()

    fig = make_subplots(
        rows=5, cols=n,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        subplot_titles=[t for t in tickers] + [""] * (4 * n),
        row_heights=[0.25, 0.20, 0.20, 0.15, 0.20],
        specs=[[{"type": "xy"}] * n] * 5,
    )

    for col_idx, ticker in enumerate(tickers, start=1):
        data      = results[ticker]
        spot      = data.get("spot", 0)
        em        = data.get("expected_move", {})
        gex       = data.get("gex", {})
        pcr       = data.get("pcr", {})
        ivr       = data.get("iv_rank", {})
        unusual   = data.get("unusual_volume", [])
        ms        = data.get("model_signal", {})
        pred      = data.get("prediction", {})
        flow_bias = data.get("flow_bias", "Neutral")

        # Row 1: price levels
        call_wall = gex.get("call_wall")
        put_wall  = gex.get("put_wall")
        upper     = em.get("upper_target", spot)
        lower     = em.get("lower_target", spot)
        levels, level_labels, level_colors = [], [], []
        if call_wall: levels.append(call_wall); level_labels.append(f"Call wall ${call_wall}"); level_colors.append(_RED)
        if upper:     levels.append(upper);     level_labels.append(f"EM upper ${upper}");       level_colors.append(_BLUE)
        if spot:      levels.append(spot);      level_labels.append(f"Spot ${spot}");            level_colors.append(_TEXT)
        if lower:     levels.append(lower);     level_labels.append(f"EM lower ${lower}");       level_colors.append(_BLUE)
        if put_wall:  levels.append(put_wall);  level_labels.append(f"Put wall ${put_wall}");    level_colors.append(_GREEN)

        fig.add_trace(go.Bar(
            x=level_labels, y=levels, marker_color=level_colors,
            text=[f"${v}" for v in levels], textposition="outside",
            textfont=dict(color=_TEXT, size=9), showlegend=False,
        ), row=1, col=col_idx)

        # Row 2: call vs put volume
        fig.add_trace(go.Bar(
            x=["Calls", "Puts"],
            y=[pcr.get("call_volume", 0), pcr.get("put_volume", 0)],
            marker_color=[_GREEN, _RED], showlegend=False,
            text=[f"{pcr.get('call_volume',0):,}", f"{pcr.get('put_volume',0):,}"],
            textposition="outside", textfont=dict(color=_TEXT, size=9),
        ), row=2, col=col_idx)

        # Row 3: IV rank
        iv_val   = ivr.get("iv_rank", 50) or 50
        iv_label = ivr.get("iv_label", "N/A")
        cur_iv   = ivr.get("current_iv", 0) or 0
        iv_color = _iv_color(iv_val, low_good=True)
        fig.add_trace(go.Bar(
            x=["IV Rank"], y=[iv_val], marker_color=[iv_color],
            text=[f"IVR {iv_val:.1f} — {iv_label[:18]}\nIV={cur_iv:.3f}"],
            textposition="outside", textfont=dict(color=_TEXT, size=9),
            showlegend=False,
        ), row=3, col=col_idx)

        # Row 4: unusual volume
        if unusual:
            top3     = unusual[:3]
            u_labels = [f"{u['type']} ${u['strike']}" for u in top3]
            u_ratios = [u["vol_oi_ratio"] for u in top3]
            u_colors = [_RED if u["type"] == "PUT" else _GREEN for u in top3]
            fig.add_trace(go.Bar(
                x=u_labels, y=u_ratios, marker_color=u_colors,
                text=[f"{r}x" for r in u_ratios], textposition="outside",
                textfont=dict(color=_TEXT, size=9), showlegend=False,
            ), row=4, col=col_idx)
        else:
            fig.add_trace(go.Bar(
                x=["No unusual volume"], y=[0], marker_color=[_SUBTEXT], showlegend=False,
            ), row=4, col=col_idx)

        # Row 5: model vs flow
        has_model = not ms.get("error") and ms.get("direction")
        has_pred  = pred.get("status") != "insufficient_data" and has_model
        if has_model:
            model_prob = ms.get("prob_up", 0.5)
            flow_score = data.get("flow_score", 0)
            bar_colors = [
                _GREEN if model_prob >= 0.5 else _RED,
                _GREEN if flow_score > 0 else _RED if flow_score < 0 else _ORANGE,
            ]
            fig.add_trace(go.Bar(
                x=["Model", "Flow"],
                y=[model_prob, min(max((flow_score + 3) / 6, 0), 1)],
                marker_color=bar_colors,
                text=[
                    f"{model_prob:.1%} {ms['direction']} ({ms.get('horizon', '?')})",
                    f"{'UP' if flow_score > 0 else 'DOWN'} score={flow_score}",
                ],
                textposition="outside", textfont=dict(color=_TEXT, size=9),
                showlegend=False,
            ), row=5, col=col_idx)
        else:
            fig.add_trace(go.Bar(
                x=["No prediction"], y=[0], marker_color=[_SUBTEXT], showlegend=False,
            ), row=5, col=col_idx)

    fig.update_layout(
        height=1000,
        title=dict(text=f"Options Flow Dashboard — {date.today()}", font=dict(color=_TEXT, size=18), x=0.5),
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(color=_TEXT), showlegend=False,
        margin=dict(t=80, b=60, l=40, r=40),
    )
    for i in range(1, 6):
        for j in range(1, n + 1):
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color=_SUBTEXT, size=8), row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, tickfont=dict(color=_SUBTEXT, size=8), row=i, col=j)

    return fig


def generate_options_dashboard(
    tickers: list | None = None,
    results: dict | None = None,
) -> str:
    """Build and save the options flow Plotly dashboard. Returns the output path."""
    if tickers is None:
        tickers = TICKERS
    if results is None:
        print(f"Scanning {len(tickers)} tickers for options flow...")
        results = scan_all(tickers)

    print("Building dashboard...")
    fig = build_dashboard(results)

    os.makedirs("charts", exist_ok=True)
    out = os.path.join("charts", "options_flow_dashboard.html")
    fig.write_html(out)
    print(f"Dashboard saved: {out}")
    return out


def open_dashboard(
    tickers: list | None = None,
    results: dict | None = None,
) -> None:
    """Generate the dashboard and open it in the default browser."""
    import webbrowser
    out      = generate_options_dashboard(tickers, results=results)
    abs_path = os.path.abspath(out)   # BUG FIX: was missing the () call
    try:
        webbrowser.open(f"file://{abs_path}")
        print(f"Opening in browser: {abs_path}")
    except Exception as e:
        print(f"Could not auto-open: {e}")
        print(f"Open manually: {abs_path}")


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else TICKERS
    results = scan_all(tickers)
    print_flow_report(results)
    open_dashboard(tickers, results=results)
