import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

from data_loader import load_all_tickers
from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from features import add_features, get_feature_columns
from splitter import time_split, make_sequences
from config import TICKERS, SEQUENCE_LENGTH, TRAIN_RATIO, VAL_RATIO

import tensorflow as tf

def load_test_data(ticker, df, macro_df=None, sentiment=None):
    sentiment_series = sentiment.get(ticker) if sentiment else None
    df        = add_features(df, macro_df=macro_df,
                             sentiment_series=sentiment_series)
    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment_series is not None
    )
    feat_cols = [c for c in feat_cols if c in df.columns]

    n       = len(df)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X = df[feat_cols].values
    y = df["target"].values

    return X[val_end:], y[val_end:], feat_cols

def evaluate_xgboost(ticker, X_test, y_test):
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    model_path  = os.path.join("models", f"{ticker}_xgb.pkl")
    if not os.path.exists(model_path):
        print(f"  No XGBoost model found for {ticker}")
        return None, None
    scaler   = joblib.load(scaler_path)
    model    = joblib.load(model_path)
    X_scaled = scaler.transform(X_test)
    probs    = model.predict_proba(X_scaled)[:, 1]
    auc      = roc_auc_score(y_test, probs)
    return probs, auc

def evaluate_lstm(ticker, X_test, y_test):
    scaler_path = os.path.join("models", f"{ticker}_lstm_scaler.pkl")
    model_path  = os.path.join("models", f"{ticker}_lstm.keras")
    if not os.path.exists(model_path):
        print(f"  No LSTM model found for {ticker}")
        return None, None
    scaler   = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_test)
    X_seq, y_seq = make_sequences(X_scaled, y_test, SEQUENCE_LENGTH)
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X_seq, verbose=0).flatten()
    auc   = roc_auc_score(y_seq, probs)
    return probs, auc

def print_confusion(y_true, y_pred_probs, model_name, ticker):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix — {model_name} on {ticker}")
    print(f"  {'':12} Predicted 0  Predicted 1")
    print(f"  Actual 0   {cm[0,0]:>10}  {cm[0,1]:>10}")
    print(f"  Actual 1   {cm[1,0]:>10}  {cm[1,1]:>10}")
    print()
    print(classification_report(y_true, y_pred,
                                 target_names=["Down", "Up"], digits=3))

def directional_accuracy_report(ticker, y_true, y_pred_probs, model_name):
    y_pred     = (y_pred_probs >= 0.5).astype(int)
    pred_up    = y_pred == 1
    pred_down  = y_pred == 0

    correct_up   = (y_true[pred_up]   == 1).sum()
    correct_down = (y_true[pred_down] == 0).sum()
    total_up     = pred_up.sum()
    total_down   = pred_down.sum()

    acc_up   = correct_up   / total_up   if total_up   > 0 else 0
    acc_down = correct_down / total_down if total_down > 0 else 0
    overall  = (correct_up + correct_down) / len(y_true)
    base     = y_true.mean()

    up_edge   = acc_up   - base
    down_edge = acc_down - (1 - base)

    print(f"\n  Directional Accuracy — {model_name} on {ticker}")
    print(f"  {'─'*50}")
    print(f"  Market base rate (actual up days):  {base:.1%}")
    print(f"  {'─'*50}")
    print(f"  Predicted UP   ({total_up:>4} signals): "
          f"{acc_up:.1%} correct  "
          f"({'BEAT' if up_edge > 0 else 'MISS'} base rate)")
    print(f"  Predicted DOWN ({total_down:>4} signals): "
          f"{acc_down:.1%} correct  "
          f"({'BEAT' if down_edge > 0 else 'MISS'} base rate)")
    print(f"  Overall accuracy:                   {overall:.1%}")
    print(f"  {'─'*50}")
    print(f"  Up signal edge:   {up_edge:+.1%}")
    print(f"  Down signal edge: {down_edge:+.1%}")

    return {
        "base_rate":  base,
        "acc_up":     acc_up,
        "acc_down":   acc_down,
        "overall":    overall,
        "total_up":   int(total_up),
        "total_down": int(total_down),
        "up_edge":    up_edge,
        "down_edge":  down_edge,
    }

def plot_roc_curves(results: dict):
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (ticker, models) in zip(axes, results.items()):
        for name, (probs, y_test) in models.items():
            if probs is None:
                continue
            RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)
        ax.set_title(ticker)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.tight_layout()
    out = os.path.join("models", "roc_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  ROC curves saved to {out}")
    plt.close()

def plot_auc_comparison(summary: dict):
    tickers   = list(summary.keys())
    xgb_aucs  = [summary[t].get("XGBoost") or 0 for t in tickers]
    lstm_aucs = [summary[t].get("LSTM")    or 0 for t in tickers]
    x     = np.arange(len(tickers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, xgb_aucs,  width, label="XGBoost", color="#4C72B0")
    ax.bar(x + width/2, lstm_aucs, width, label="LSTM",    color="#DD8452")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Test AUC")
    ax.set_title("XGBoost vs LSTM — Test AUC per ticker")
    ax.legend()
    ax.set_ylim(0.45, 0.65)
    plt.tight_layout()
    out = os.path.join("models", "auc_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  AUC comparison saved to {out}")
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()

    summary     = {}
    dir_summary = {}
    roc_data    = {}

    print(f"\n{'═'*60}")
    print("  EVALUATION REPORT")
    print(f"{'═'*60}")

    for ticker, df in all_data.items():
        print(f"\n{'─'*40}")
        print(f"  Ticker: {ticker}")
        print(f"{'─'*40}")

        X_test, y_test, _ = load_test_data(
            ticker, df, macro_df=macro_df, sentiment=sentiment)

        print(f"  Test set: {len(y_test)} samples | "
              f"Up: {y_test.sum()} | Down: {(1-y_test).sum()}")

        xgb_probs, xgb_auc = evaluate_xgboost(ticker, X_test, y_test)
        if xgb_probs is not None:
            print(f"\n  XGBoost Test AUC: {xgb_auc:.4f}")
            print_confusion(y_test, xgb_probs, "XGBoost", ticker)
            xgb_dir = directional_accuracy_report(
                ticker, y_test, xgb_probs, "XGBoost")
        else:
            xgb_dir = {}

        lstm_probs, lstm_auc = evaluate_lstm(ticker, X_test, y_test)
        if lstm_probs is not None:
            y_seq = y_test[SEQUENCE_LENGTH:]
            print(f"\n  LSTM Test AUC: {lstm_auc:.4f}")
            print_confusion(y_seq, lstm_probs, "LSTM", ticker)
            lstm_dir = directional_accuracy_report(
                ticker, y_seq, lstm_probs, "LSTM")
        else:
            lstm_dir = {}

        summary[ticker] = {
            "XGBoost": xgb_auc if xgb_probs is not None else None,
            "LSTM":    lstm_auc if lstm_probs is not None else None,
        }
        dir_summary[ticker] = {
            "XGBoost": xgb_dir,
            "LSTM":    lstm_dir,
        }
        roc_data[ticker] = {
            "XGBoost": (xgb_probs,  y_test) if xgb_probs  is not None else (None, None),
            "LSTM":    (lstm_probs, y_test[SEQUENCE_LENGTH:]) if lstm_probs is not None else (None, None),
        }

    # AUC summary
    print(f"\n{'═'*60}")
    print("  AUC SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<8} {'XGBoost':>10} {'LSTM':>10}")
    print(f"  {'─'*32}")
    for ticker, scores in summary.items():
        xgb  = f"{scores['XGBoost']:.4f}" if scores["XGBoost"] else "N/A"
        lstm = f"{scores['LSTM']:.4f}"    if scores["LSTM"]    else "N/A"
        print(f"  {ticker:<8} {xgb:>10} {lstm:>10}")

    # Directional accuracy summary
    print(f"\n{'═'*60}")
    print("  DIRECTIONAL ACCURACY SUMMARY (XGBoost)")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<8} {'Base%':>7} {'Up acc':>8} "
          f"{'Dn acc':>8} {'Overall':>9} {'Up edge':>9}")
    print(f"  {'─'*55}")
    for ticker, scores in dir_summary.items():
        xgb = scores.get("XGBoost", {})
        if xgb:
            print(
                f"  {ticker:<8} "
                f"{xgb['base_rate']:>7.1%} "
                f"{xgb['acc_up']:>8.1%} "
                f"{xgb['acc_down']:>8.1%} "
                f"{xgb['overall']:>9.1%} "
                f"{xgb['up_edge']:>+9.1%}"
            )

    print()
    plot_roc_curves(roc_data)
    plot_auc_comparison(summary)
    print("\nEvaluation complete. Charts saved to models/ folder.")