# app/ml/__init__.py
# Re-exports the public API of the ML pipeline so callers can do:
#
#   from app.ml import add_features, get_feature_columns, time_split
#   from app.ml import run_predictions, apply_entry_filters
#
# Training scripts (train_classical, train_lstm, tune, evaluate) are not
# re-exported here because they are run as standalone scripts, not imported
# by the application at runtime.

from .features import add_features, get_feature_columns
from .splitter import time_split, make_sequences
from .predict import (
    run_predictions,
    get_today_signal,
    apply_entry_filters,
    fetch_latest_data,
    TICKER_WIN_RATES,
)
