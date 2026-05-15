"""
Trainer — trains models and decides whether to accept or reject them.

Trainer takes a pre-built feature matrix and target series from
FeatureEngineer, trains two models (XGBoost and LogisticRegression),
evaluates them on a time-based holdout set, selects the better one,
and compares its AUC against the previously accepted model.

A retrain is accepted if:
    auc_new >= auc_previous + auc_improvement_threshold

A retrain is rejected if:
    auc_new < auc_previous + auc_improvement_threshold

Rejection is a valid, expected outcome — not an error. Trainer returns
a TrainingResult with accepted=False. TrainingError is only raised for
genuine failures: OOM, corrupted data that passed validation, etc.

Why time-based train/holdout split:
    Stock return data is a time series. Using sklearn's random train_test_split
    would shuffle the data, causing future rows to appear in the training set
    and past rows to appear in the holdout set. This produces optimistically
    biased AUC scores that don't reflect real-world performance.

    Instead we split chronologically:
        Train set:   first (1 - holdout_fraction) of rows
        Holdout set: last holdout_fraction of rows

    This simulates the real scenario: train on historical data, evaluate on
    more recent data you didn't train on.

Models trained:
    XGBoost     — gradient boosted trees. Handles nonlinear feature interactions
                  well. GPU-accelerated if CUDA is available.
    LogReg      — logistic regression. Fast, interpretable baseline. Good when
                  the signal is weak or data is limited.

    The model with the higher holdout AUC is selected as the winner.
    Only the winning model is saved to ConfigStore.

Depends on:
    ConfigStore         — writes the accepted TickerConfig
    ModelAuditRepository — reads previous AUC, writes audit record
    ConfigRepository    — reads training hyperparameters from SYSTEM_CONFIG
    TrainingResult      — returned to caller
    TrainingError       — raised for genuine failures only

Exposes:
    train(symbol, feature_df, feat_cols, target, fwd_days) → TrainingResult
"""

from __future__ import annotations

import pickle
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.config_store import ConfigStore
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.model_audit_repository import ModelAuditRepository
from src.data.database import ModelAuditModel
from src.utils.exceptions import TrainingError
from src.utils.types import ConfigCategory, TickerConfig, TrainingResult

import pandas as pd


class Trainer:
    """
    Trains XGBoost and LogisticRegression models and manages the
    accept/reject decision based on AUC improvement.

    Usage:
        trainer = Trainer(config_store, audit_repo, config_repo)
        result = trainer.train(
            symbol="AAPL",
            feature_df=feature_df,
            feat_cols=feat_cols,
            target=target,
            fwd_days=5,
        )

        if result.accepted:
            # New model is live — ConfigStore has been updated
        else:
            # Old model remains active — result.rejection_reason explains why
    """

    def __init__(
        self,
        config_store: ConfigStore,
        audit_repo: ModelAuditRepository,
        config_repo: ConfigRepository,
    ) -> None:
        """
        Args:
            config_store: For writing the accepted TickerConfig to disk.
            audit_repo: For reading previous AUC and writing audit records.
            config_repo: For reading training hyperparameters.
        """
        self._config_store = config_store
        self._audit_repo = audit_repo
        self._config_repo = config_repo
        self._use_gpu = torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        feat_cols: list[str],
        target: pd.Series,
        fwd_days: int,
    ) -> TrainingResult:
        """
        Train models on the provided feature matrix and evaluate them.

        Steps:
            1. Read hyperparameters from SYSTEM_CONFIG.
            2. Time-based train/holdout split.
            3. Scale features with StandardScaler (fit on train only).
            4. Train XGBoost and LogisticRegression.
            5. Evaluate both on holdout, pick the one with higher AUC.
            6. Read previous accepted AUC from ModelAuditRepository.
            7. Compare new AUC to previous AUC + improvement threshold.
            8. Accept or reject based on comparison.
            9. Write TickerConfig to ConfigStore (accepted or rejected path).
            10. Write ModelAudit row.
            11. Return TrainingResult.

        Args:
            symbol: The ticker being trained.
            feature_df: Feature matrix from FeatureEngineer.build_train_matrix().
            feat_cols: Ordered feature column names — written into TickerConfig.
            target: Binary target series (0/1) aligned with feature_df.
            fwd_days: Forward horizon this model predicts.

        Returns:
            TrainingResult with accepted, auc_before, auc_after,
            n_train_rows, rejection_reason, config_path, trained_at.

        Raises:
            TrainingError: Only for genuine failures (OOM, corrupted data).
                           NOT raised for model rejection.
        """
        try:
            return self._run_training(
                symbol, feature_df, feat_cols, target, fwd_days
            )
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(
                f"Training failed unexpectedly for {symbol}: {exc}",
                cause=exc,
            )

    # ------------------------------------------------------------------
    # Private — training pipeline
    # ------------------------------------------------------------------

    def _run_training(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        feat_cols: list[str],
        target: pd.Series,
        fwd_days: int,
    ) -> TrainingResult:
        """
        Inner training pipeline — separated from train() so TrainingError
        wrapping in train() only catches unexpected exceptions.
        """
        # Step 1 — read hyperparameters
        holdout_fraction = self._config_repo.get(
            ConfigCategory.TRAINING, "holdout_fraction"
        )
        auc_threshold = self._config_repo.get(
            ConfigCategory.TRAINING, "auc_improvement_threshold"
        )
        xgb_n_estimators = self._config_repo.get(
            ConfigCategory.TRAINING, "xgb_n_estimators"
        )
        xgb_max_depth = self._config_repo.get(
            ConfigCategory.TRAINING, "xgb_max_depth"
        )
        xgb_learning_rate = self._config_repo.get(
            ConfigCategory.TRAINING, "xgb_learning_rate"
        )
        logreg_max_iter = self._config_repo.get(
            ConfigCategory.TRAINING, "logreg_max_iter"
        )

        # Step 2 — time-based split
        n_total = len(feature_df)
        n_train = int(n_total * (1 - holdout_fraction))

        if n_train < 50:
            raise TrainingError(
                f"Not enough training rows for {symbol}: "
                f"{n_train} after split. Need at least 50."
            )

        X = feature_df[feat_cols].values
        y = target.values

        X_train, X_holdout = X[:n_train], X[n_train:]
        y_train, y_holdout = y[:n_train], y[n_train:]

        if len(X_holdout) < 10:
            raise TrainingError(
                f"Holdout set for {symbol} has only {len(X_holdout)} rows. "
                f"Need at least 10 for a reliable AUC estimate."
            )

        # Step 3 — scale features
        # Fit scaler ONLY on train set. Transform both train and holdout.
        # Fitting on the full dataset would leak holdout statistics into
        # the training process — a form of data leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_holdout_scaled = scaler.transform(X_holdout)

        # Step 4 — train both models
        xgb_model, xgb_auc = self._train_xgboost(
            X_train_scaled, y_train,
            X_holdout_scaled, y_holdout,
            xgb_n_estimators, xgb_max_depth, xgb_learning_rate,
        )

        logreg_model, logreg_auc = self._train_logreg(
            X_train_scaled, y_train,
            X_holdout_scaled, y_holdout,
            logreg_max_iter,
        )

        # Step 5 — pick the winning model
        if xgb_auc >= logreg_auc:
            winning_model = xgb_model
            winning_auc = xgb_auc
            winning_name = "xgboost"
        else:
            winning_model = logreg_model
            winning_auc = logreg_auc
            winning_name = "logreg"

        # Step 6 — get the previous AUC
        previous_audit = self._audit_repo.get_latest_accepted(symbol)
        auc_before = previous_audit.auc_after if previous_audit else None

        # Step 7 — accept/reject decision
        if auc_before is None:
            # First train — always accept
            accepted = True
            rejection_reason = None
        elif winning_auc >= auc_before + auc_threshold:
            accepted = True
            rejection_reason = None
        else:
            accepted = False
            rejection_reason = (
                f"New model AUC ({winning_auc:.4f}) did not improve over "
                f"previous AUC ({auc_before:.4f}) by the required margin "
                f"of {auc_threshold:.4f}. "
                f"Previous model remains active."
            )

        # Step 8 — build TickerConfig
        ticker_config = TickerConfig(
            symbol=symbol,
            schema_version=ConfigStore.CURRENT_SCHEMA_VERSION,
            feat_cols=feat_cols,
            fwd_days=fwd_days,
            model_name=winning_name,
            auc=winning_auc,
            trained_at=datetime.utcnow(),
            macro_symbols=[
                col.replace("macro_", "").replace("_ret1", "")
                for col in feat_cols
                if col.startswith("macro_")
            ],
            extra={
                "scaler": scaler,
                "model": winning_model,
                "n_train_rows": n_train,
                "xgb_auc": xgb_auc,
                "logreg_auc": logreg_auc,
            },
        )

        # Step 9 — write config to disk
        if accepted:
            config_path = self._config_store.save(ticker_config)
        else:
            config_path = self._config_store.save_rejected(ticker_config)

        # Step 10 — write audit record
        audit_row = ModelAuditModel(
            ticker=symbol,
            auc_before=auc_before,
            auc_after=winning_auc,
            accepted=accepted,
            config_path=config_path,
            n_train_rows=n_train,
            rejection_reason=rejection_reason,
            trained_at=datetime.utcnow(),
        )
        self._audit_repo.add(audit_row)

        # Step 11 — return result
        return TrainingResult(
            symbol=symbol,
            accepted=accepted,
            auc_before=auc_before,
            auc_after=winning_auc,
            n_train_rows=n_train,
            rejection_reason=rejection_reason,
            config_path=config_path,
            trained_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Private — model training helpers
    # ------------------------------------------------------------------

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> tuple[XGBClassifier, float]:
        """
        Train an XGBoost classifier and evaluate its holdout AUC.

        Uses GPU acceleration if CUDA is available (RTX 4080 Super).
        Falls back to CPU if CUDA is not available.

        Args:
            X_train: Scaled training features.
            y_train: Training labels.
            X_holdout: Scaled holdout features.
            y_holdout: Holdout labels.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Step size shrinkage.

        Returns:
            Tuple of (fitted XGBClassifier, holdout AUC score).
        """
        device = "cuda" if self._use_gpu else "cpu"

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            device=device,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_holdout, y_holdout)],
            verbose=False,
        )

        proba = model.predict_proba(X_holdout)[:, 1]
        auc = float(roc_auc_score(y_holdout, proba))

        return model, auc

    def _train_logreg(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        max_iter: int,
    ) -> tuple[LogisticRegression, float]:
        """
        Train a LogisticRegression classifier and evaluate its holdout AUC.

        LogisticRegression is the interpretable baseline. It performs well
        when the relationship between features and returns is roughly linear,
        which is common in low-volatility, large-cap tickers.

        Args:
            X_train: Scaled training features.
            y_train: Training labels.
            X_holdout: Scaled holdout features.
            y_holdout: Holdout labels.
            max_iter: Maximum iterations for the solver to converge.

        Returns:
            Tuple of (fitted LogisticRegression, holdout AUC score).
        """
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            solver="lbfgs",
            C=1.0,
        )

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_holdout)[:, 1]
        auc = float(roc_auc_score(y_holdout, proba))

        return model, auc