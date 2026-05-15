"""
TickerOnboardingPipeline — full onboarding workflow for a new ticker.

Coordinates the complete sequence of steps required to bring a new
ticker from PENDING to COMPLETE status:

    PENDING → FETCHING:
        1. TickerValidator.validate() — confirm symbol exists and is eligible.
           If invalid: transition to FAILED, return result.
           If ineligible: record ticker, set status appropriately, return result.

    FETCHING → ANALYZING:
        2. DataLoader.fetch_full() — download complete price history.
        3. DataLoader.fetch_macro() — download all macro indicator histories.
           If fetch fails: transition to FAILED, return result.

    ANALYZING → TRAINING:
        4. MacroCorrelationAnalyzer.analyze() — select relevant macro indicators.
           If analysis fails: transition to FAILED, return result.

    TRAINING → COMPLETE or FAILED:
        5. FeatureEngineer.build_train_matrix() — build the feature matrix.
        6. Trainer.train() — train models, evaluate AUC, accept or reject.
           If first-train rejected: transition to FAILED (no baseline model exists).
           If accepted: transition to COMPLETE.

run() never raises. All failures are caught, the ticker status is
transitioned to FAILED with a reason, and an OnboardingResult with
success=False is returned.

Depends on: all domain components and data layer components.

Exposes:
    run(symbol) → OnboardingResult
"""

from __future__ import annotations

from datetime import datetime

from src.data.config_store import ConfigStore
from src.data.database import TickerModel
from src.data.market_cache import MarketCache
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.macro_repository import MacroRepository
from src.data.repositories.ticker_repository import TickerRepository
from src.domain.ml_pipeline.data_loader import DataLoader
from src.domain.ml_pipeline.feature_engineer import FeatureEngineer
from src.domain.ml_pipeline.macro_correlation_analyzer import MacroCorrelationAnalyzer
from src.domain.ml_pipeline.trainer import Trainer
from src.domain.onboarding.ticker_validator import TickerValidator
from src.utils.exceptions import StockPredictorError
from src.utils.types import (
    ConfigCategory,
    OnboardingResult,
    OnboardingStatus,
)


class TickerOnboardingPipeline:
    """
    Executes the full onboarding workflow for a new ticker.

    Usage:
        pipeline = TickerOnboardingPipeline(...)
        result = pipeline.run("NVDA")

        if result.success:
            print(f"NVDA is ready — model trained and config saved.")
        else:
            print(f"Onboarding failed: {result.reason}")
    """

    def __init__(
        self,
        ticker_repo: TickerRepository,
        macro_repo: MacroRepository,
        config_repo: ConfigRepository,
        validator: TickerValidator,
        data_loader: DataLoader,
        macro_analyzer: MacroCorrelationAnalyzer,
        feature_engineer: FeatureEngineer,
        trainer: Trainer,
        session,
    ) -> None:
        self._ticker_repo = ticker_repo
        self._macro_repo = macro_repo
        self._config_repo = config_repo
        self._validator = validator
        self._data_loader = data_loader
        self._macro_analyzer = macro_analyzer
        self._feature_engineer = feature_engineer
        self._trainer = trainer
        self._session = session

    def run(self, symbol: str) -> OnboardingResult:
        """
        Execute the full onboarding sequence for one ticker.

        Creates the ticker row if it doesn't exist, then steps through
        the state machine from PENDING to COMPLETE or FAILED.

        Args:
            symbol: The ticker symbol to onboard, e.g. "NVDA".

        Returns:
            OnboardingResult with success, final_status, and reason.
            Never raises — all failures produce success=False.
        """
        # Ensure ticker row exists
        self._ensure_ticker_exists(symbol)

        # Step 1 — Validate
        self._set_status(symbol, OnboardingStatus.FETCHING)
        validation = self._validator.validate(symbol)

        if not validation.is_valid:
            return self._fail(symbol, validation.reason or "Ticker validation failed.")

        if not validation.is_eligible:
            self._ticker_repo.update_training_flags(
                symbol, training_eligible=False, min_data_met=False
            )
            self._session.commit()
            return OnboardingResult(
                symbol=symbol,
                success=False,
                final_status=OnboardingStatus.FETCHING,
                reason=validation.reason,
            )

        # Step 2 — Full data fetch
        try:
            self._data_loader.fetch_full(symbol)
            self._fetch_all_macros()
        except StockPredictorError as exc:
            return self._fail(symbol, str(exc))

        # Step 3 — Macro correlation analysis
        self._set_status(symbol, OnboardingStatus.ANALYZING)
        try:
            profile = self._macro_analyzer.analyze(symbol)
            self._session.commit()
        except StockPredictorError as exc:
            return self._fail(symbol, str(exc))

        # Step 4 — Feature engineering
        self._set_status(symbol, OnboardingStatus.TRAINING)
        try:
            fwd_days = self._config_repo.get(
                ConfigCategory.TRAINING, "default_fwd_days"
            )
            feature_df, feat_cols, target = self._feature_engineer.build_train_matrix(
                symbol=symbol,
                macro_symbols=profile.selected_macros,
                fwd_days=fwd_days,
            )
        except StockPredictorError as exc:
            return self._fail(symbol, str(exc))

        # Step 5 — Train
        try:
            result = self._trainer.train(
                symbol=symbol,
                feature_df=feature_df,
                feat_cols=feat_cols,
                target=target,
                fwd_days=fwd_days,
            )
        except StockPredictorError as exc:
            return self._fail(symbol, str(exc))

        # First-train rejection means no model exists — treat as failure
        if not result.accepted:
            return self._fail(
                symbol,
                f"First training run rejected: {result.rejection_reason}"
            )

        # Success
        self._ticker_repo.update_training_flags(
            symbol, training_eligible=True, min_data_met=True
        )
        self._set_status(symbol, OnboardingStatus.COMPLETE)
        self._session.commit()

        return OnboardingResult(
            symbol=symbol,
            success=True,
            final_status=OnboardingStatus.COMPLETE,
            reason=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_ticker_exists(self, symbol: str) -> None:
        """Create a ticker row if one does not already exist."""
        if not self._ticker_repo.exists(symbol):
            ticker = TickerModel(
                symbol=symbol,
                onboarding_status=OnboardingStatus.PENDING.value,
                added_at=datetime.utcnow(),
            )
            self._ticker_repo.add(ticker)
            self._session.commit()

    def _set_status(self, symbol: str, status: OnboardingStatus) -> None:
        """Transition onboarding status and commit."""
        self._ticker_repo.update_onboarding_status(symbol, status)
        self._session.commit()

    def _fail(self, symbol: str, reason: str) -> OnboardingResult:
        """Transition to FAILED and return a failure result."""
        self._ticker_repo.update_onboarding_status(
            symbol, OnboardingStatus.FAILED
        )
        self._session.commit()
        return OnboardingResult(
            symbol=symbol,
            success=False,
            final_status=OnboardingStatus.FAILED,
            reason=reason,
        )

    def _fetch_all_macros(self) -> None:
        """Fetch data for every macro indicator in the candidate universe."""
        indicators = self._macro_repo.get_all_indicators()
        for indicator in indicators:
            try:
                self._data_loader.fetch_macro(indicator.symbol)
            except StockPredictorError:
                # Macro fetch failure is non-fatal — the correlation analyzer
                # handles missing macro data by scoring it 0.0
                pass