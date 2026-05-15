"""
InferencePipeline — assembles and executes one full inference run for a ticker.

InferencePipeline is the top-level coordinator of the inference flow.
It pulls together every component needed to go from "ticker symbol" to
"filtered signal with audit trail" in one method call.

Inference flow for one ticker:
    1. Load TickerConfig from ConfigStore (raises ConfigNotFoundError if missing)
    2. Run DataLoader.fetch_incremental() to update the market cache
    3. Call FeatureEngineer.build_inference_matrix() with feat_cols from config
    4. Load the trained model and scaler from config.extra
    5. Scale features with the stored scaler
    6. Call model.predict_proba() to get the raw signal probabilities
    7. Determine the raw signal (highest probability class)
    8. Assemble MarketContext (VIX value, RSI, earnings flag)
    9. Call SignalFilter.filter() to apply gates
    10. Return PredictionResult

InferencePipeline never writes to the database. It never writes to disk.
InferenceOrchestrator (application layer) decides what to persist.

Error propagation:
    ConfigNotFoundError and DataFetchError propagate unchanged — callers
    need to know the specific failure type.
    All other unexpected errors are wrapped in InferenceError.

Depends on:
    ConfigStore         — loads TickerConfig
    DataLoader          — updates market cache
    FeatureEngineer     — builds inference feature matrix
    SignalFilter        — applies market condition gates
    ConfigRepository    — reads signal gate configuration
    EarningsRepository  — checks for upcoming earnings events

Exposes:
    predict(symbol) → PredictionResult
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from src.data.config_store import ConfigStore
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.earnings_repository import EarningsRepository
from src.domain.ml_pipeline.data_loader import DataLoader
from src.domain.ml_pipeline.feature_engineer import FeatureEngineer
from src.domain.ml_pipeline.signal_filter import SignalFilter
from src.utils.exceptions import (
    ConfigNotFoundError,
    DataFetchError,
    InferenceError,
)
from src.utils.types import (
    ConfigCategory,
    FilterResult,
    MarketContext,
    PredictionResult,
    SignalValue,
)


# Mapping from model class index to SignalValue.
# XGBoost and LogReg are trained on binary targets (0=negative, 1=positive).
# We map those back to BUY/SELL and use HOLD only via SignalFilter suppression.
_CLASS_TO_SIGNAL = {
    0: SignalValue.SELL,
    1: SignalValue.BUY,
}


class InferencePipeline:
    """
    Executes a full inference run for one ticker and returns a PredictionResult.

    Usage:
        pipeline = InferencePipeline(
            config_store, data_loader, feature_engineer,
            signal_filter, config_repo, earnings_repo
        )
        result = pipeline.predict("AAPL")
        # result.signal is BUY, HOLD, or SELL
        # result.filter_result shows which gates fired
    """

    def __init__(
        self,
        config_store: ConfigStore,
        data_loader: DataLoader,
        feature_engineer: FeatureEngineer,
        signal_filter: SignalFilter,
        config_repo: ConfigRepository,
        earnings_repo: EarningsRepository,
    ) -> None:
        self._config_store = config_store
        self._data_loader = data_loader
        self._feature_engineer = feature_engineer
        self._signal_filter = signal_filter
        self._config_repo = config_repo
        self._earnings_repo = earnings_repo

    def predict(self, symbol: str) -> PredictionResult:
        """
        Run a full inference cycle for one ticker.

        Args:
            symbol: The ticker symbol to run inference for.

        Returns:
            PredictionResult with signal, probability, fwd_days, model_name,
            filter_result, and predicted_at.

        Raises:
            ConfigNotFoundError: If no trained model exists for this ticker.
            DataFetchError: If the incremental market data fetch fails.
            InferenceError: For any other unexpected failure during inference.
        """
        try:
            return self._run(symbol)
        except (ConfigNotFoundError, DataFetchError):
            # Let these propagate unchanged — InferenceOrchestrator handles them
            raise
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(
                f"Inference failed unexpectedly for {symbol}: {exc}",
                cause=exc,
            )

    # ------------------------------------------------------------------
    # Private — inference pipeline steps
    # ------------------------------------------------------------------

    def _run(self, symbol: str) -> PredictionResult:
        """
        Inner inference pipeline. Separated from predict() so the
        exception wrapping in predict() stays clean.
        """
        # Step 1 — load config (raises ConfigNotFoundError if missing)
        config = self._config_store.load(symbol)

        # Step 2 — update market cache with latest data
        self._data_loader.fetch_incremental(symbol)

        # Step 3 — build inference feature matrix using feat_cols from config
        feature_df = self._feature_engineer.build_inference_matrix(
            symbol=symbol,
            macro_symbols=config.macro_symbols,
            feat_cols=config.feat_cols,
        )

        # Step 4 — load model and scaler from config.extra
        model = config.extra.get("model")
        scaler = config.extra.get("scaler")

        if model is None or scaler is None:
            raise InferenceError(
                f"Config for {symbol} is missing model or scaler in "
                f"config.extra. The model may have been saved incorrectly. "
                f"Retrain {symbol} to fix this."
            )

        # Step 5 — scale features using the stored scaler
        X = feature_df.values
        X_scaled = scaler.transform(X)

        # Step 6 — get raw probabilities from the model
        proba = model.predict_proba(X_scaled)[0]
        # proba is [prob_class_0, prob_class_1] = [prob_SELL, prob_BUY]

        # Step 7 — determine raw signal and probability
        predicted_class = int(np.argmax(proba))
        raw_signal = _CLASS_TO_SIGNAL[predicted_class]
        raw_probability = float(proba[predicted_class])

        # Step 8 — assemble MarketContext for SignalFilter
        context = self._build_market_context(symbol, feature_df)

        # Step 9 — apply gates
        gate_config = self._config_repo.get_all_for_category(ConfigCategory.SIGNALS)
        filter_result = self._signal_filter.filter(
            raw_signal=raw_signal,
            probability=raw_probability,
            context=context,
            config=gate_config,
        )

        # Step 10 — return result
        return PredictionResult(
            symbol=symbol,
            signal=filter_result.signal,
            probability=filter_result.probability,
            fwd_days=config.fwd_days,
            model_name=config.model_name,
            filter_result=filter_result,
            predicted_at=datetime.utcnow(),
        )

    def _build_market_context(
        self,
        symbol: str,
        feature_df,
    ) -> MarketContext:
        """
        Assemble a MarketContext from the computed feature row.

        Extracts VIX value and RSI value directly from the feature matrix
        (they are already computed as features). Checks EarningsRepository
        for upcoming earnings within the configured blackout window.

        Args:
            symbol: The ticker symbol.
            feature_df: The single-row inference feature DataFrame.

        Returns:
            A populated MarketContext ready for SignalFilter.
        """
        row = feature_df.iloc[0]

        # Extract VIX from macro feature if present
        vix_col = "macro_^VIX_ret1"
        vix_value = float(row[vix_col]) if vix_col in feature_df.columns else None

        # Extract RSI from price features
        rsi_value = float(row["rsi_14"]) if "rsi_14" in feature_df.columns else None

        # Determine price trend from SMA crossover feature
        price_trend = None
        if "sma_cross" in feature_df.columns:
            sma_cross = float(row["sma_cross"])
            if sma_cross > 1.02:
                price_trend = "up"
            elif sma_cross < 0.98:
                price_trend = "down"
            else:
                price_trend = "flat"

        # Check for upcoming earnings within the blackout window
        blackout_days = self._config_repo.get(
            ConfigCategory.SIGNALS,
            "earnings_blackout_days",
        )
        upcoming = self._earnings_repo.get_upcoming_for_ticker(symbol)
        has_upcoming_earnings = (
            upcoming is not None
            and upcoming.report_date <= datetime.utcnow() + timedelta(days=blackout_days)
        )

        return MarketContext(
            symbol=symbol,
            vix_value=vix_value,
            rsi_value=rsi_value,
            price_trend=price_trend,
            has_upcoming_earnings=has_upcoming_earnings,
        )