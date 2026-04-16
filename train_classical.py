import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from config import TRAIN_RATIO, VAL_RATIO

from data_loader import load_all_tickers
from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from features import add_features, get_feature_columns
from splitter import time_split

def train_ticker(ticker: str, df, macro_df=None, sentiment=None):
    print(f"\n{'─'*40}")
    print(f"  Ticker: {ticker}")
    print(f"{'─'*40}")

    sentiment_series = sentiment.get(ticker) if sentiment else None
    df        = add_features(df, macro_df=macro_df,
                             sentiment_series=sentiment_series)
    feat_cols = get_feature_columns(
        include_macro=macro_df is not None,
        include_sentiment=sentiment_series is not None
    )
    feat_cols = [c for c in feat_cols if c in df.columns]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(df, feat_cols)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    results = {}

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results["LogReg"] = {
        "val_auc":  roc_auc_score(y_val,  lr.predict_proba(X_val)[:,1]),
        "test_auc": roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]),
    }

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    results["XGBoost"] = {
        "val_auc":  roc_auc_score(y_val,  xgb.predict_proba(X_val)[:,1]),
        "test_auc": roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]),
    }

    joblib.dump(scaler, os.path.join("models", f"{ticker}_scaler.pkl"))
    joblib.dump(xgb,    os.path.join("models", f"{ticker}_xgb.pkl"))

    print()
    for name, scores in results.items():
        print(f"  {name:<12} Val AUC: {scores['val_auc']:.4f}  |  Test AUC: {scores['test_auc']:.4f}")

    return results

def train_combined(all_data, macro_df=None, sentiment=None):
    print(f"\n{'═'*40}")
    print("  COMBINED MODEL — all tickers")
    print(f"{'═'*40}")

    ticker_list = list(all_data.keys())
    train_frames, val_frames, test_frames = [], [], []

    for ticker, df in all_data.items():
        sentiment_series = sentiment.get(ticker) if sentiment else None
        df_feat = add_features(df.copy(), macro_df=macro_df,
                               sentiment_series=sentiment_series)
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
        include_sentiment=sentiment is not None
    ) + ["ticker_id"]
    feat_cols = [c for c in feat_cols if c in train_df.columns]

    X_train = train_df[feat_cols].values
    y_train = train_df["target"].values
    X_val   = val_df[feat_cols].values
    y_val   = val_df["target"].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["target"].values

    print(f"  Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_auc  = roc_auc_score(y_val,  xgb.predict_proba(X_val)[:,1])
    test_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])

    print(f"\n  Combined XGBoost | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    joblib.dump(scaler, os.path.join("models", "combined_scaler.pkl"))
    joblib.dump(xgb,    os.path.join("models", "combined_xgb.pkl"))

    return val_auc, test_auc

if __name__ == "__main__":
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()

    all_results = {}
    for ticker, df in all_data.items():
        all_results[ticker] = train_ticker(ticker, df,
                                           macro_df=macro_df,
                                           sentiment=sentiment)

    print(f"\n{'═'*50}")
    print("  SUMMARY — XGBoost Test AUC per ticker")
    print(f"{'═'*50}")
    for ticker, res in all_results.items():
        print(f"  {ticker:<6} {res['XGBoost']['test_auc']:.4f}")

    train_combined(all_data, macro_df=macro_df, sentiment=sentiment)