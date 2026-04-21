# options_flow.py
# Options flow scanner using yfinance options chains.
# Computes PCR, unusual volume, GEX, IV rank, expected move,
# and combines with ML model signal for price direction prediction.

import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import date, datetime
from config import TICKERS

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_chain(ticker: str, expiry: str) -> tuple:
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception as e:
        print(f"  Could not fetch chain for {ticker} {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def get_spot_price(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).info.get("regularMarketPrice", 0))
    except Exception:
        return 0.0

def get_nearest_expiries(ticker: str, n: int = 4) -> list:
    try:
        return list(yf.Ticker(ticker).options[:n])
    except Exception:
        return []

# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_pcr(calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """
    Put/Call Ratio by volume and open interest.
    PCR < 0.7  = bullish sentiment
    PCR > 1.2  = bearish / heavy hedging
    """
    call_vol = calls["volume"].fillna(0).sum()
    put_vol  = puts["volume"].fillna(0).sum()
    call_oi  = calls["openInterest"].fillna(0).sum()
    put_oi   = puts["openInterest"].fillna(0).sum()

    pcr_vol = put_vol / (call_vol + 1e-9)
    pcr_oi  = put_oi  / (call_oi  + 1e-9)

    if pcr_vol < 0.7:
        sentiment = "Bullish"
    elif pcr_vol > 1.2:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"

    return {
        "pcr_volume":  round(pcr_vol, 3),
        "pcr_oi":      round(pcr_oi, 3),
        "call_volume": int(call_vol),
        "put_volume":  int(put_vol),
        "call_oi":     int(call_oi),
        "put_oi":      int(put_oi),
        "sentiment":   sentiment,
    }

def find_unusual_volume(calls: pd.DataFrame,
                         puts: pd.DataFrame,
                         threshold: float = 2.0) -> pd.DataFrame:
    """
    Flags contracts where volume > threshold * open interest.
    Filters out zero-OI contracts to avoid division artifacts.
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
        unusual = df[df["vol_oi_ratio"] >= threshold].copy()

        for _, row in unusual.iterrows():
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

    result = pd.DataFrame(records)
    return result.sort_values("vol_oi_ratio", ascending=False).reset_index(drop=True)

def compute_gex(calls: pd.DataFrame,
                puts: pd.DataFrame,
                spot: float) -> dict:
    """
    Gamma Exposure — estimates net gamma of market makers.
    Filters to strikes within 15% of spot to avoid deep OTM noise.

    Positive GEX = price dampening (pinning near call wall)
    Negative GEX = price amplifying (trending, breakout likely)
    """
    if calls.empty and puts.empty:
        return {"total_gex": 0, "call_wall": None,
                "put_wall": None, "gex_bias": "Unknown", "spot": spot}

    call_gex = calls[
        (calls["strike"] >= spot * 0.85) &
        (calls["strike"] <= spot * 1.15) &
        (calls["openInterest"] > 0) &
        (calls["impliedVolatility"] > 0.01)
    ].copy()

    put_gex = puts[
        (puts["strike"] >= spot * 0.85) &
        (puts["strike"] <= spot * 1.15) &
        (puts["openInterest"] > 0) &
        (puts["impliedVolatility"] > 0.01)
    ].copy()

    if call_gex.empty and put_gex.empty:
        return {"total_gex": 0, "call_wall": None,
                "put_wall": None, "gex_bias": "Unknown", "spot": spot}

    call_gex["gex"] = (
        call_gex["openInterest"] *
        call_gex["impliedVolatility"] *
        (spot ** 2) * 0.01
    )
    put_gex["gex"] = -(
        put_gex["openInterest"] *
        put_gex["impliedVolatility"] *
        (spot ** 2) * 0.01
    )

    total_gex       = call_gex["gex"].sum() + put_gex["gex"].sum()
    max_call_strike = (
        call_gex.loc[call_gex["gex"].idxmax(), "strike"]
        if not call_gex.empty else None
    )
    max_put_strike  = (
        put_gex.loc[put_gex["gex"].idxmin(), "strike"]
        if not put_gex.empty else None
    )
    gex_bias = "Pinning" if total_gex > 0 else "Trending"

    return {
        "total_gex":  round(total_gex, 2),
        "call_wall":  max_call_strike,
        "put_wall":   max_put_strike,
        "gex_bias":   gex_bias,
        "spot":       spot,
    }

def compute_iv_rank(ticker: str) -> dict:
    """
    IV Rank — where current IV sits relative to recent range.
    Skips first expiry (same-day IV unreliable), uses next 3.
    Filters to near-the-money strikes only.

    IVR < 30 = cheap options (good to buy)
    IVR > 70 = expensive options (good to sell)
    """
    try:
        stock    = yf.Ticker(ticker)
        expiries = stock.options
        if not expiries:
            return {"iv_rank": None, "current_iv": None, "iv_label": "Unknown"}

        spot    = get_spot_price(ticker)
        iv_list = []

        for exp in expiries[1:4]:
            calls, _ = fetch_chain(ticker, exp)
            if calls.empty:
                continue
            calls = calls[
                (calls["strike"] >= spot * 0.95) &
                (calls["strike"] <= spot * 1.05) &
                (calls["impliedVolatility"] > 0.01)
            ]
            if calls.empty:
                continue
            calls         = calls.copy()
            calls["dist"] = (calls["strike"] - spot).abs()
            atm           = calls.loc[calls["dist"].idxmin()]
            iv            = float(atm["impliedVolatility"])
            if iv > 0.01:
                iv_list.append(iv)

        if not iv_list:
            return {"iv_rank": None, "current_iv": None, "iv_label": "Unknown"}

        current_iv = np.mean(iv_list)
        iv_min     = min(iv_list)
        iv_max     = max(iv_list)
        iv_rank    = (current_iv - iv_min) / (iv_max - iv_min + 1e-9) * 100

        if iv_rank < 30:
            iv_label = "Low (cheap options)"
        elif iv_rank < 70:
            iv_label = "Normal"
        else:
            iv_label = "High (expensive options)"

        return {
            "iv_rank":    round(iv_rank, 1),
            "current_iv": round(current_iv, 4),
            "iv_label":   iv_label,
        }
    except Exception as e:
        return {"iv_rank": None, "current_iv": None, "iv_label": f"Error: {e}"}

def compute_expected_move(spot: float,
                           iv: float,
                           days_to_expiry: int) -> dict:
    """
    Expected Move — market-implied 1 std dev price range by expiry.
    Formula: EM = spot * IV * sqrt(days / 365)
    ~68% probability price stays within this range.
    """
    if not iv or not spot or iv < 0.01:
        return {}

    em     = spot * iv * np.sqrt(days_to_expiry / 365)
    upper  = spot + em
    lower  = spot - em
    em_pct = (em / spot) * 100

    return {
        "expected_move":     round(em, 2),
        "expected_move_pct": round(em_pct, 2),
        "upper_target":      round(upper, 2),
        "lower_target":      round(lower, 2),
        "days_to_expiry":    days_to_expiry,
    }

# ── Model Signal ──────────────────────────────────────────────────────────────

def get_model_signal(ticker: str) -> dict:
    """
    Pulls the ML model's directional signal for a ticker.
    Loads the trained model and runs it on the latest data.
    Returns prob_up, direction, and the model name.
    """
    ticker      = ticker.upper()
    model_path  = os.path.join("models", f"{ticker}_model.pkl")
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    config_path = os.path.join("models", f"{ticker}_config.pkl")

    if not os.path.exists(model_path):
        return {"error": f"No model found for {ticker}"}

    try:
        from macro_loader import fetch_macro
        from sentiment_loader import load_all_sentiment
        from features import add_features
        from predict import fetch_latest_data

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

        has_earnings = any(
            "eps" in c or "pead" in c or "earnings" in c
            for c in feat_cols
        )
        from earnings_loader import build_earnings_features
        earn = build_earnings_features(ticker, df) if has_earnings else None

        df_feat = add_features(df, macro_df=macro_df,
                               sentiment_series=sent,
                               earnings_df=earn,
                               forward_days=fwd_days,
                               predict_mode=True)

        if df_feat.empty:
            return {"error": "Could not generate features"}

        feat_cols_present = [c for c in feat_cols if c in df_feat.columns]
        X        = df_feat[feat_cols_present].iloc[[-1]].values
        prob_up  = float(model.predict_proba(scaler.transform(X))[0][1])
        direction = "UP" if prob_up >= 0.5 else "DOWN"

        horizon_map = {1: "1d", 21: "21d", 63: "63d", 126: "126d"}
        horizon     = horizon_map.get(fwd_days, f"{fwd_days}d")

        return {
            "prob_up":   round(prob_up, 4),
            "direction": direction,
            "model":     model_name,
            "horizon":   horizon,
        }
    except Exception as e:
        return {"error": str(e)}

# ── Prediction Layer ──────────────────────────────────────────────────────────

def generate_prediction(model_signal: dict,
                         flow_bias: str,
                         expected_move: dict,
                         spot: float) -> dict:
    """
    Combines ML model signal and options flow bias into a prediction.
    Shows both signals separately so you can make your own judgment.

    Agreement = stronger conviction.
    Disagreement = conflicted, exercise caution.
    """
    if "error" in model_signal or not expected_move:
        return {"status": "insufficient_data"}

    prob_up       = model_signal["prob_up"]
    model_dir     = model_signal["direction"]
    horizon       = model_signal["horizon"]

    # Determine if model and flow agree
    flow_is_bullish  = flow_bias == "Bullish"
    model_is_bullish = model_dir == "UP"
    agreement        = flow_is_bullish == model_is_bullish

    # Price targets based on expected move
    if model_is_bullish:
        price_target = expected_move.get("upper_target")
        target_label = "UP target"
    else:
        price_target = expected_move.get("lower_target")
        target_label = "DOWN target"

    move_pct = expected_move.get("expected_move_pct", 0)
    dte      = expected_move.get("days_to_expiry", 0)

    if agreement:
        if abs(prob_up - 0.5) > 0.1:
            conviction = "HIGH"
        else:
            conviction = "MODERATE"
    else:
        conviction = "LOW — model and flow disagree"

    return {
        "model_direction":  model_dir,
        "model_prob_up":    prob_up,
        "model_horizon":    horizon,
        "flow_direction":   "UP" if flow_is_bullish else "DOWN",
        "flow_bias":        flow_bias,
        "agreement":        agreement,
        "conviction":       conviction,
        "price_target":     price_target,
        "target_label":     target_label,
        "move_pct":         move_pct,
        "dte":              dte,
        "spot":             spot,
    }

# ── Full Scanner ──────────────────────────────────────────────────────────────

def scan_ticker(ticker: str) -> dict:
    """Full options flow scan for a single ticker."""
    ticker = ticker.upper()
    print(f"  Scanning {ticker}...")

    expiries = get_nearest_expiries(ticker, n=4)
    if not expiries:
        return {"ticker": ticker, "error": "No options available"}

    spot        = get_spot_price(ticker)
    nearest     = expiries[1] if len(expiries) > 1 else expiries[0]
    calls, puts = fetch_chain(ticker, nearest)

    exp_date = datetime.strptime(nearest, "%Y-%m-%d").date()
    dte      = max((exp_date - date.today()).days, 1)

    pcr          = compute_pcr(calls, puts)
    unusual      = find_unusual_volume(calls, puts, threshold=2.0)
    gex          = compute_gex(calls, puts, spot)
    iv_rank      = compute_iv_rank(ticker)
    em           = compute_expected_move(spot, iv_rank.get("current_iv", 0), dte)
    model_signal = get_model_signal(ticker)

    # Flow score
    flow_score = 0
    if pcr["pcr_volume"] < 0.7:
        flow_score += 1
    elif pcr["pcr_volume"] > 1.2:
        flow_score -= 1
    if not unusual.empty:
        flow_score += len(unusual[unusual["type"] == "CALL"]) * 0.5
        flow_score -= len(unusual[unusual["type"] == "PUT"])  * 0.5

    flow_bias  = "Bullish" if flow_score > 0 else ("Bearish" if flow_score < 0 else "Neutral")
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

def scan_all(tickers: list = None) -> dict:
    if tickers is None:
        tickers = TICKERS
    results = {}
    for ticker in tickers:
        results[ticker] = scan_ticker(ticker)
    return results

def print_flow_report(results: dict):
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

        # Model signal
        if "error" not in ms:
            model_arrow = "▲" if ms["direction"] == "UP" else "▼"
            print(f"  MODEL  {model_arrow} {ms['direction']:<6} "
                  f"prob_up={ms['prob_up']:.1%}  "
                  f"horizon={ms['horizon']}  "
                  f"({ms['model']})")
        else:
            print(f"  MODEL  N/A — {ms.get('error', 'unknown')}")

        # Flow signal
        flow_arrow = "▲" if data["flow_bias"] == "Bullish" else ("▼" if data["flow_bias"] == "Bearish" else "─")
        print(f"  FLOW   {flow_arrow} {data['flow_bias']:<6} "
              f"PCR={pcr['pcr_volume']:.3f}  "
              f"IVR={ivr.get('iv_rank', 'N/A')}  "
              f"{ivr.get('iv_label', '')}")

        # GEX
        print(f"  GEX    {gex.get('gex_bias', 'N/A'):<20} "
              f"Call wall=${gex.get('call_wall', 'N/A')}  "
              f"Put wall=${gex.get('put_wall', 'N/A')}")

        # Expected move + price target
        if em:
            print(f"  MOVE   ±${em['expected_move']} ({em['expected_move_pct']}%)  "
                  f"→  DOWN: ${em['lower_target']}  |  UP: ${em['upper_target']}")

        # Combined prediction
        if pred.get("status") != "insufficient_data":
            agree_str = "✓ AGREE" if pred["agreement"] else "✗ DISAGREE"
            print(f"  {'─'*68}")
            print(f"  PREDICTION  {agree_str}  |  Conviction: {pred['conviction']}")
            print(f"    Model says {pred['model_direction']} ({pred['model_prob_up']:.1%}) "
                  f"over {pred['model_horizon']}")
            print(f"    Flow  says {pred['flow_direction']} (PCR {pcr['pcr_volume']:.3f})")
            if pred["price_target"]:
                print(f"    {pred['target_label']}: ${pred['price_target']} "
                      f"({pred['move_pct']}% move in {pred['dte']}d)")

        # Unusual volume
        if data["unusual_volume"]:
            print(f"  UNUSUAL  {len(data['unusual_volume'])} contracts flagged:")
            for u in data["unusual_volume"][:3]:
                print(f"    {u['type']} ${u['strike']} — "
                      f"vol={u['volume']:,} vs OI={u['open_interest']:,} "
                      f"({u['vol_oi_ratio']}x)  IV={u['iv']:.3f}")

    print(f"\n{'═'*70}\n")

if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    results = scan_all(tickers)
    print_flow_report(results)