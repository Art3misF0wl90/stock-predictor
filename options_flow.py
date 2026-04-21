# options_flow.py
# Options flow scanner using yfinance options chains.
# Computes PCR, unusual volume, GEX, IV rank, and expected move.
# Designed to integrate with predict.py signals.

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from config import TICKERS

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_chain(ticker: str, expiry: str) -> tuple:
    """Fetches calls and puts for a given ticker and expiry date."""
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception as e:
        print(f"  Could not fetch chain for {ticker} {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def get_spot_price(ticker: str) -> float:
    """Gets the current spot price of the underlying."""
    try:
        return float(yf.Ticker(ticker).info.get("regularMarketPrice", 0))
    except Exception:
        return 0.0

def get_nearest_expiries(ticker: str, n: int = 4) -> list:
    """Returns the next n expiry dates for a ticker."""
    try:
        return list(yf.Ticker(ticker).options[:n])
    except Exception:
        return []

# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_pcr(calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """
    Put/Call Ratio by both volume and open interest.
    PCR < 0.7  = bullish sentiment
    PCR > 1.2  = bearish / heavy hedging
    Extreme readings are contrarian signals.
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
    threshold=2.0 means volume is 2x the open interest.
    """
    records = []

    for df, kind in [(calls, "CALL"), (puts, "PUT")]:
        if df.empty:
            continue
        df = df.copy()
        df["volume"]       = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)

        # Only consider contracts with actual open interest
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
    Gamma Exposure (GEX) — estimates net gamma exposure of market makers.
    Filters to strikes within 15% of spot to avoid deep OTM noise.

    Positive GEX = dealers net long gamma = price dampening (pinning)
    Negative GEX = dealers net short gamma = price amplifying (trending)
    """
    if calls.empty and puts.empty:
        return {"total_gex": 0, "call_wall": None,
                "put_wall": None, "gex_bias": "Unknown", "spot": spot}

    # Filter to near-the-money strikes only
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

    total_gex = call_gex["gex"].sum() + put_gex["gex"].sum()

    max_call_strike = (
        call_gex.loc[call_gex["gex"].idxmax(), "strike"]
        if not call_gex.empty else None
    )
    max_put_strike = (
        put_gex.loc[put_gex["gex"].idxmin(), "strike"]
        if not put_gex.empty else None
    )

    gex_bias = "Pinning (low vol expected)" if total_gex > 0 else "Trending (high vol expected)"

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

    IVR < 30  = cheap options, good to buy
    IVR > 70  = expensive options, good to sell / avoid buying
    """
    try:
        stock    = yf.Ticker(ticker)
        expiries = stock.options
        if not expiries:
            return {"iv_rank": None, "current_iv": None, "iv_label": "Unknown"}

        spot    = get_spot_price(ticker)
        iv_list = []

        # Skip first expiry, use next 3
        for exp in expiries[1:4]:
            calls, puts = fetch_chain(ticker, exp)
            if calls.empty:
                continue

            # Filter to near-the-money strikes only
            calls = calls[
                (calls["strike"] >= spot * 0.95) &
                (calls["strike"] <= spot * 1.05) &
                (calls["impliedVolatility"] > 0.01)
            ]
            if calls.empty:
                continue

            calls = calls.copy()
            calls["dist"] = (calls["strike"] - spot).abs()
            atm = calls.loc[calls["dist"].idxmin()]
            iv  = float(atm["impliedVolatility"])
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
    Expected Move — market-implied 1 standard deviation price range by expiry.
    Formula: EM = spot * IV * sqrt(days / 365)
    Gives the range where market expects price with ~68% probability.
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

# ── Full Scanner ──────────────────────────────────────────────────────────────

def scan_ticker(ticker: str) -> dict:
    """
    Full options flow scan for a single ticker.
    Returns PCR, unusual volume, GEX, IV rank, and expected move.
    """
    print(f"  Scanning {ticker}...")
    stock    = yf.Ticker(ticker)
    expiries = get_nearest_expiries(ticker, n=4)

    if not expiries:
        return {"ticker": ticker, "error": "No options available"}

    spot = get_spot_price(ticker)

    # Use second expiry — first is often same-day with bad IV data
    nearest     = expiries[1] if len(expiries) > 1 else expiries[0]
    calls, puts = fetch_chain(ticker, nearest)

    # Days to expiry
    exp_date = datetime.strptime(nearest, "%Y-%m-%d").date()
    dte      = max((exp_date - date.today()).days, 1)

    # Core metrics
    pcr     = compute_pcr(calls, puts)
    unusual = find_unusual_volume(calls, puts, threshold=2.0)
    gex     = compute_gex(calls, puts, spot)
    iv_rank = compute_iv_rank(ticker)
    em      = compute_expected_move(
        spot,
        iv_rank.get("current_iv", 0),
        dte
    )

    # Flow score
    flow_score = 0
    if pcr["pcr_volume"] < 0.7:
        flow_score += 1
    elif pcr["pcr_volume"] > 1.2:
        flow_score -= 1

    if not unusual.empty:
        call_unusual = unusual[unusual["type"] == "CALL"]
        put_unusual  = unusual[unusual["type"] == "PUT"]
        flow_score  += len(call_unusual) * 0.5
        flow_score  -= len(put_unusual)  * 0.5

    if flow_score > 0:
        flow_bias = "Bullish"
    elif flow_score < 0:
        flow_bias = "Bearish"
    else:
        flow_bias = "Neutral"

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
    }

def scan_all(tickers: list = None) -> dict:
    """Scans all tickers and returns combined flow report."""
    if tickers is None:
        tickers = TICKERS
    results = {}
    for ticker in tickers:
        results[ticker] = scan_ticker(ticker)
    return results

def print_flow_report(results: dict):
    """Prints a formatted flow report to the terminal."""
    print(f"\n{'═'*65}")
    print(f"  OPTIONS FLOW REPORT — {date.today()}")
    print(f"{'═'*65}")
    print(f"  {'Ticker':<8} {'Bias':<10} {'PCR':<8} {'IVR':<8} "
          f"{'IV Label':<25} {'Call Wall':<12} {'Put Wall'}")
    print(f"  {'─'*65}")

    for ticker, data in results.items():
        if "error" in data:
            print(f"  {ticker:<8} ERROR: {data['error']}")
            continue

        pcr = data["pcr"]
        gex = data["gex"]
        ivr = data["iv_rank"]

        print(f"  {data['ticker']:<8} {data['flow_bias']:<10} "
              f"{pcr['pcr_volume']:<8.3f} "
              f"{str(ivr.get('iv_rank', 'N/A')):<8} "
              f"{ivr.get('iv_label', 'N/A'):<25} "
              f"{str(gex.get('call_wall', 'N/A')):<12} "
              f"{gex.get('put_wall', 'N/A')}")

        em = data.get("expected_move", {})
        if em:
            print(f"  {'':8} Expected move: ±${em['expected_move']} "
                  f"({em['expected_move_pct']}%) "
                  f"→ ${em['lower_target']} to ${em['upper_target']}")

        if data["unusual_volume"]:
            print(f"  {'':8} Unusual volume ({len(data['unusual_volume'])} contracts):")
            for u in data["unusual_volume"][:3]:
                print(f"  {'':10} {u['type']} ${u['strike']} — "
                      f"vol={u['volume']} vs OI={u['open_interest']} "
                      f"({u['vol_oi_ratio']}x)")
        print()

if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    results = scan_all(tickers)
    print_flow_report(results)