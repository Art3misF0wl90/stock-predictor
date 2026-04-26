# app/ml/tune.py
#
# Optional hyperparameter search for XGBoost using Optuna.
#
# Runs 150 Bayesian optimisation trials per ticker, searching over:
#   n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
#   min_child_weight, gamma, reg_alpha, reg_lambda
#
# Results are saved to models/best_params.pkl.
# To use them in training, load the file in train_classical.py and pass
# the params dict to XGBClassifier(**params).
#
# WARNING: this script is slow — expect 1-3 hours for the full watchlist.
# Skip it on first run; the default XGBoost params in train_classical.py
# are already reasonable.

import os

import joblib
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import TICKERS
from app.data import load_all_tickers
from app.data import load_all_earnings
from app.ml.features import add_features, get_feature_columns
from app.data import fetch_macro
from app.data import load_all_sentiment
from app.ml.splitter import time_split

TICKERS_NO_EARNINGS = ["TSLA", "SPY"]


def tune_ticker(
    ticker: str,
    df,
    macro_df,
    sentiment,
    earnings,
    n_trials: int = 150,
) -> tuple[dict, float]:
    """
    Run Optuna hyperparameter search for one ticker.

    Returns (best_params, best_val_auc).
    """
    print(f"\nTuning {ticker} ({n_trials} trials)...")

    sentiment_series = sentiment.get(ticker) if sentiment else None
    earnings_df = (
        None if ticker in TICKERS_NO_EARNINGS
        else (earnings.get(ticker) if earnings else None)
    )

    df_feat   = add_features(
        df,
        macro_df=macro_df,
        sentiment_series=sentiment_series,
        earnings_df=earnings_df,
    )
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

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
            "eval_metric":      "logloss",
            "random_state":     42,
            "verbosity":        0,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best val AUC: {study.best_value:.4f}")
    print("  Best params:")
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
            ticker, df, macro_df, sentiment, earnings, n_trials=150
        )
        all_best[ticker] = params

    print(f"\n{'='*60}")
    print("  TUNING COMPLETE — best params per ticker")
    print(f"{'='*60}")
    for ticker, params in all_best.items():
        print(f"\n{ticker}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(all_best, os.path.join("models", "best_params.pkl"))
    print("\nSaved to models/best_params.pkl")
