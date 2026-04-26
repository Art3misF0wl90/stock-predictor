# app/ml/train_classical.py
#
# Trains XGBoost and Logistic Regression classifiers for every ticker.
#
# For each ticker the script tries every combination of:
#   forward horizon  — 1d, 21d, 63d  (from FORWARD_DAYS_LIST in config.py)
#   model type       — LogisticRegression, XGBoost
#
# The combination with the highest test-set AUC-ROC is saved as the
# "best" model for that ticker under models/<TICKER>_model.pkl.
#
# A combined model is also trained on all tickers stacked together,
# with a ticker_id integer as an additional feature.  This is used by
# analyze.py for tickers that do not have a ticker-specific model.
#
# Artefacts written to models/:
#   <TICKER>_model.pkl    — best model object
#   <TICKER>_scaler.pkl   — fitted StandardScaler
#   <TICKER>_config.pkl   — dict with fwd_days, model_name, feat_cols, test_auc
#   best_overall.pkl      — dict mapping ticker → best config (for inspection)
#   combined_xgb.pkl      — combined XGBoost model
#   combined_scaler.pkl   — combined scaler

import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import TRAIN_RATIO, VAL_RATIO, FORWARD_DAYS_LIST, FORWARD_DAYS
from app.data import load_all_tickers
from app.data import load_all_earnings
from app.ml.features import add_features, get_feature_columns
from app.data import fetch_macro
from app.data import load_all_sentiment
from app.ml.splitter import time_split

# Tickers excluded from earnings feature engineering (missing / unreliable data)
TICKERS_NO_EARNINGS = ["TSLA", "SPY"]


def train_combined(
    all_data: dict,
    macro_df=None,
    sentiment=None,
    earnings=None,
) -> tuple[float, float]:
    """
    Train a single XGBoost on all tickers stacked together.

    A ticker_id integer column is added so the model can distinguish
    per-ticker behaviour without one-hot encoding.
    """
    print(f"\n{'═'*40}")
    print("  COMBINED MODEL — all tickers")
    print(f"{'═'*40}")

    ticker_list = list(all_data.keys())
    train_frames, val_frames, test_frames = [], [], []

    for ticker, df in all_data.items():
        sentiment_series = sentiment.get(ticker) if sentiment else None
        earnings_df = (
            None if ticker in TICKERS_NO_EARNINGS
            else (earnings.get(ticker) if earnings else None)
        )
        df_feat = add_features(
            df.copy(),
            macro_df=macro_df,
            sentiment_series=sentiment_series,
            earnings_df=earnings_df,
            forward_days=FORWARD_DAYS,
        )
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
        include_earnings=True,
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

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_auc  = roc_auc_score(y_val,  xgb.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"\n  Combined XGBoost | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "combined_scaler.pkl"))
    joblib.dump(xgb,    os.path.join("models", "combined_xgb.pkl"))

    return val_auc, test_auc


if __name__ == "__main__":
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()
    earnings  = load_all_earnings(all_data)

    best_overall = {}
    os.makedirs("models", exist_ok=True)

    for ticker, df in all_data.items():
        print(f"\n{'═'*50}")
        print(f"  {ticker} — searching best horizon + model")
        print(f"{'═'*50}")

        sentiment_series = sentiment.get(ticker) if sentiment else None
        earnings_df = (
            None if ticker in TICKERS_NO_EARNINGS
            else (earnings.get(ticker) if earnings else None)
        )

        best_test_auc = 0.0
        best_config   = None

        for fwd in FORWARD_DAYS_LIST:
            df_feat = add_features(
                df.copy(),
                macro_df=macro_df,
                sentiment_series=sentiment_series,
                earnings_df=earnings_df,
                forward_days=fwd,
            )

            feat_cols = get_feature_columns(
                include_macro=True,
                include_sentiment=True,
                include_earnings=ticker not in TICKERS_NO_EARNINGS,
            )
            feat_cols = [c for c in feat_cols if c in df_feat.columns]

            (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(
                df_feat, feat_cols
            )

            scaler    = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X_val)
            X_test_s  = scaler.transform(X_test)

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_s, y_train)
            lr_test = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

            xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            xgb.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            xgb_test = roc_auc_score(y_test, xgb.predict_proba(X_test_s)[:, 1])

            print(f"  {fwd:>3}d | LogReg: {lr_test:.4f} | XGBoost: {xgb_test:.4f}")

            for model, name, test_auc in [
                (lr,  "LogReg",  lr_test),
                (xgb, "XGBoost", xgb_test),
            ]:
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_config   = {
                        "model":      model,
                        "scaler":     scaler,
                        "model_name": name,
                        "fwd_days":   fwd,
                        "test_auc":   test_auc,
                        "feat_cols":  feat_cols,
                    }

        print(
            f"\n  BEST → {best_config['model_name']} "
            f"{best_config['fwd_days']}d "
            f"Test AUC: {best_config['test_auc']:.4f}"
        )

        joblib.dump(best_config["scaler"], os.path.join("models", f"{ticker}_scaler.pkl"))
        joblib.dump(best_config["model"],  os.path.join("models", f"{ticker}_model.pkl"))
        joblib.dump(best_config,           os.path.join("models", f"{ticker}_config.pkl"))
        best_overall[ticker] = best_config

    print(f"\n{'═'*50}")
    print("  FINAL BEST PER TICKER")
    print(f"{'═'*50}")
    print(f"  {'Ticker':<8} {'Model':<10} {'Horizon':>8} {'Test AUC':>10}")
    print(f"  {'─'*40}")
    for ticker, cfg in best_overall.items():
        print(
            f"  {ticker:<8} {cfg['model_name']:<10} "
            f"{cfg['fwd_days']:>6}d {cfg['test_auc']:>10.4f}"
        )

    joblib.dump(best_overall, os.path.join("models", "best_overall.pkl"))
    print("\nSaved best_overall.pkl")

    train_combined(all_data, macro_df=macro_df, sentiment=sentiment, earnings=earnings)
