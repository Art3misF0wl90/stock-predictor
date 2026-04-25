import optuna
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from data_loader import load_all_tickers
from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from earnings_loader import load_all_earnings
from features import add_features, get_feature_columns
from splitter import time_split
from config import TICKERS

TICKERS_NO_EARNINGS = ["TSLA", "SPY"]

def tune_ticker(ticker, df, macro_df, sentiment, earnings, n_trials=150):
    print(f"\nTuning {ticker} ({n_trials} trials)...")

    sentiment_series = sentiment.get(ticker) if sentiment else None
    earnings_df      = (
        None if ticker in TICKERS_NO_EARNINGS
        else (earnings.get(ticker) if earnings else None)
    )

    df_feat   = add_features(df, macro_df=macro_df,
                             sentiment_series=sentiment_series,
                             earnings_df=earnings_df)
    feat_cols = get_feature_columns(
        include_macro=True,
        include_sentiment=True,
        include_earnings=ticker not in TICKERS_NO_EARNINGS,
    )
    feat_cols = [c for c in feat_cols if c in df_feat.columns]

    (X_train, y_train), (X_val, y_val), _ = time_split(df_feat, feat_cols)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
            "eval_metric":      "logloss",
            "random_state":     42,
            "verbosity":        0,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:,1])

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best val AUC: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params, study.best_value

if __name__ == "__main__":
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()
    earnings  = load_all_earnings(all_data)

    all_best = {}
    for ticker, df in all_data.items():
        params, val_auc = tune_ticker(
            ticker, df, macro_df, sentiment, earnings, n_trials=150)
        all_best[ticker] = params

    print(f"\n{'='*60}")
    print("  TUNING COMPLETE — best params per ticker")
    print(f"{'='*60}")
    for ticker, params in all_best.items():
        print(f"\n{ticker}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    # Save best params for use in train_classical.py
    joblib.dump(all_best, os.path.join("models", "best_params.pkl"))
    print("\nSaved to models/best_params.pkl")