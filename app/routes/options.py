# app/routes/options.py
#
# Blueprint for the options flow scanner.
# Manages the background scan cache and exposes the scan trigger,
# results read, and chart endpoints.
#
# The in-memory cache (_options_cache) is module-level here rather
# than on app.py so it stays co-located with the code that uses it.

import threading
from datetime import datetime

from flask import Blueprint, jsonify, render_template, Response

options_bp = Blueprint("options", __name__)

# In-memory cache shared across requests within this process.
_options_cache: dict = {
    "results": None,
    "timestamp": None,
    "scanning": False,
}


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@options_bp.route("/options-flow")
def options_flow_page():
    return render_template("options_flow.html")


# ---------------------------------------------------------------------------
# Read cached results
# ---------------------------------------------------------------------------

@options_bp.route("/api/options-flow")
def api_options_flow():
    return jsonify({
        "scanning": _options_cache["scanning"],
        "has_data": _options_cache["results"] is not None,
        "results": _options_cache["results"] or {},
        "timestamp": _options_cache["timestamp"],
    })


# ---------------------------------------------------------------------------
# Trigger a new scan (runs in a background thread)
# ---------------------------------------------------------------------------

@options_bp.route("/api/options-flow/scan", methods=["POST"])
def api_options_flow_scan():
    if _options_cache["scanning"]:
        return jsonify({"status": "already_scanning"})

    def _scan():
        _options_cache["scanning"] = True
        try:
            from app.services.options_flow import scan_all
            from config import TICKERS
            _options_cache["results"] = scan_all(TICKERS)
            _options_cache["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Options flow scan error: {e}")
        finally:
            _options_cache["scanning"] = False

    threading.Thread(target=_scan, daemon=True).start()
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# Chart (regenerates from cached results if available)
# ---------------------------------------------------------------------------

@options_bp.route("/api/options-flow/chart")
def api_options_flow_chart():
    import os
    chart_path = os.path.join("charts", "options_flow_dashboard.html")

    if _options_cache.get("results"):
        from app.services.options_flow import generate_options_dashboard
        generate_options_dashboard(results=_options_cache["results"])

    if not os.path.exists(chart_path):
        return "No chart available — run a scan first.", 404

    with open(chart_path, "r", encoding="utf-8") as f:
        html = f.read()
    return Response(html, content_type="text/html")