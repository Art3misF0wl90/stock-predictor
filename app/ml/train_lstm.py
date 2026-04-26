# app/ml/train_lstm.py
#
# Trains stacked LSTM networks per ticker and a combined model.
#
# Architecture (per-ticker):
#   Input → LSTM(64, return_sequences=True) → Dropout(0.3) → BatchNorm
#         → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1, sigmoid)
#
# The combined model uses LSTM(128) and LSTM(64) to handle the larger
# cross-ticker dataset.
#
# In practice, per-ticker XGBoost models tend to match or beat LSTM AUC
# on this dataset size.  The LSTM models are evaluated alongside classical
# models in evaluate.py; train_classical.py picks the winner per ticker.
#
# Artefacts written to models/:
#   <TICKER>_lstm.keras       — saved Keras model
#   <TICKER>_lstm_scaler.pkl  — fitted StandardScaler
#   combined_lstm.keras
#   combined_lstm_scaler.pkl

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import RANDOM_SEED, SEQUENCE_LENGTH, TRAIN_RATIO, VAL_RATIO
from app.data import load_all_tickers
from app.ml.features import add_features, get_feature_columns
from app.data import fetch_macro
from app.data import load_all_sentiment
from app.ml.splitter import make_sequences, time_split

tf.random.set_seed(RANDOM_SEED)


def build_lstm(input_shape: tuple) -> Sequential:
    """Build and compile the per-ticker LSTM architecture."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit_callbacks() -> list:
    """Standard callbacks used by both per-ticker and combined training."""
    return [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", patience=5,
            factor=0.5, verbose=1,
        ),
    ]


def train_ticker_lstm(
    ticker: str,
    df: pd.DataFrame,
    macro_df=None,
    sentiment=None,
) -> tuple[object, float, float]:
    """
    Train a per-ticker LSTM and save artefacts to models/.

    Returns (history, val_auc, test_auc).
    """
    print(f"\n{'─'*40}")
    print(f"  LSTM — Ticker: {ticker}")
    print(f"{'─'*40}")

    sentiment_series = sentiment.get(ticker) if sentiment else None
    df = add_features(df, macro_df=macro_df, sentiment_series=sentiment_series)
    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment_series is not None,
    )
    feat_cols = [c for c in feat_cols if c in df.columns]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(df, feat_cols)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    X_train_seq, y_train_seq = make_sequences(X_train, y_train, SEQUENCE_LENGTH)
    X_val_seq,   y_val_seq   = make_sequences(X_val,   y_val,   SEQUENCE_LENGTH)
    X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  SEQUENCE_LENGTH)

    print(
        f"  Shapes → Train: {X_train_seq.shape} "
        f"| Val: {X_val_seq.shape} "
        f"| Test: {X_test_seq.shape}"
    )

    model = build_lstm(input_shape=(SEQUENCE_LENGTH, X_train_seq.shape[2]))
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=32,
        callbacks=_fit_callbacks(),
        verbose=1,
    )

    val_auc  = roc_auc_score(y_val_seq,  model.predict(X_val_seq,  verbose=0).flatten())
    test_auc = roc_auc_score(y_test_seq, model.predict(X_test_seq, verbose=0).flatten())
    print(f"\n  LSTM | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", f"{ticker}_lstm.keras"))
    joblib.dump(scaler, os.path.join("models", f"{ticker}_lstm_scaler.pkl"))

    return model.history, val_auc, test_auc


def train_combined_lstm(all_data: dict, macro_df=None, sentiment=None) -> tuple[float, float]:
    """Train a single larger LSTM on all tickers stacked together."""
    print(f"\n{'='*40}")
    print("  COMBINED LSTM — all tickers")
    print(f"{'='*40}")

    ticker_list = list(all_data.keys())
    train_frames, val_frames, test_frames = [], [], []

    for ticker, df in all_data.items():
        sentiment_series = sentiment.get(ticker) if sentiment else None
        df_feat = add_features(df.copy(), macro_df=macro_df, sentiment_series=sentiment_series)
        df_feat["ticker_id"] = ticker_list.index(ticker)

        n         = len(df_feat)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_frames.append(df_feat.iloc[:train_end])
        val_frames.append(df_feat.iloc[train_end:val_end])
        test_frames.append(df_feat.iloc[val_end:])

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df   = pd.concat(val_frames,   ignore_index=True)
    test_df  = pd.concat(test_frames,  ignore_index=True)

    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment is not None,
    ) + ["ticker_id"]
    feat_cols = [c for c in feat_cols if c in train_df.columns]

    X_train, y_train = train_df[feat_cols].values, train_df["target"].values
    X_val,   y_val   = val_df[feat_cols].values,   val_df["target"].values
    X_test,  y_test  = test_df[feat_cols].values,  test_df["target"].values

    print(f"  Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    X_train_seq, y_train_seq = make_sequences(X_train, y_train, SEQUENCE_LENGTH)
    X_val_seq,   y_val_seq   = make_sequences(X_val,   y_val,   SEQUENCE_LENGTH)
    X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  SEQUENCE_LENGTH)

    print(
        f"  Shapes → Train: {X_train_seq.shape} "
        f"| Val: {X_val_seq.shape} "
        f"| Test: {X_test_seq.shape}"
    )

    # Larger capacity for the cross-ticker combined model
    model = Sequential([
        LSTM(128, input_shape=(SEQUENCE_LENGTH, X_train_seq.shape[2]), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=64,
        callbacks=_fit_callbacks(),
        verbose=1,
    )

    val_auc  = roc_auc_score(y_val_seq,  model.predict(X_val_seq,  verbose=0).flatten())
    test_auc = roc_auc_score(y_test_seq, model.predict(X_test_seq, verbose=0).flatten())
    print(f"\n  Combined LSTM | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "combined_lstm.keras"))
    joblib.dump(scaler, os.path.join("models", "combined_lstm_scaler.pkl"))

    return val_auc, test_auc


if __name__ == "__main__":
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()
    summary   = {}

    for ticker, df in all_data.items():
        _, val_auc, test_auc = train_ticker_lstm(
            ticker, df, macro_df=macro_df, sentiment=sentiment
        )
        summary[ticker] = {"val_auc": val_auc, "test_auc": test_auc}

    print(f"\n{'='*50}")
    print("  SUMMARY — LSTM Test AUC per ticker")
    print(f"{'='*50}")
    for ticker, scores in summary.items():
        print(f"  {ticker:<6} {scores['test_auc']:.4f}")

    train_combined_lstm(all_data, macro_df=macro_df, sentiment=sentiment)
