# app/services/__init__.py
# Re-exports the public API of each service so callers can do:
#
#   from app.services import analyze_ticker, get_todays_signals, chat
#
# Heavy modules (bot, options_flow, backtest) are imported lazily at call
# time inside their own modules to avoid slow startup — they are NOT
# re-exported here for that reason.

from .database import (
    get_todays_signals,
    get_signal_history,
    get_actionable_signals,
    get_summary_stats,
    get_recent_performance,
    record_outcome,
)
from .analyze import analyze_ticker, add_ticker_to_watchlist
