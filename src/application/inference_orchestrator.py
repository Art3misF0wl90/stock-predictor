"""
InferenceOrchestrator — runs inference for all tickers and handles results.

For each ticker on the watchlist:
    1. InferencePipeline.predict() — get PredictionResult
    2. Persist signal to SignalRepository
    3. AlertEvaluator.evaluate() — generate alert candidates
    4. AlertDeduplicator.filter() — remove duplicates
    5. Persist unique alerts to AlertRepository
    6. SuggestionEngine.generate() — generate suggestion if inputs changed
    7. Persist suggestion to SuggestionRepository if new

run_for_all() iterates every watchlist ticker and collects results.
run_for_ticker() handles one ticker — used by the scheduled job and
by RetrainOrchestrator after an accepted retrain.

Exposes:
    run_for_all()       → list[InferenceOrchestratorResult]
    run_for_ticker(sym) → InferenceOrchestratorResult
"""

from __future__ import annotations

from datetime import datetime

from src.data.database import AlertModel, SignalModel, SuggestionLogModel
from src.data.repositories.alert_repository import AlertRepository
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.earnings_repository import EarningsRepository
from src.data.repositories.sentiment_repository import SentimentRepository
from src.data.repositories.signal_repository import SignalRepository
from src.data.repositories.suggestion_repository import SuggestionRepository
from src.data.repositories.ticker_repository import TickerRepository
from src.data.repositories.transaction_repository import TransactionRepository
from src.domain.ml_pipeline.inference_pipeline import InferencePipeline
from src.domain.suggestions.alert_deduplicator import AlertDeduplicator
from src.domain.suggestions.alert_evaluator import AlertEvaluator, AlertEvaluatorInputs
from src.domain.suggestions.suggestion_engine import SuggestionEngine, SuggestionInputs
from src.utils.exceptions import StockPredictorError
from src.utils.types import (
    AlertType,
    ConfigCategory,
    InferenceOrchestratorResult,
    PositionDirection,
    SignalValue,
)


class InferenceOrchestrator:
    """
    Coordinates inference, alert generation, and suggestion generation.

    Usage:
        orchestrator = InferenceOrchestrator(...)
        results = orchestrator.run_for_all()
    """

    def __init__(
        self,
        ticker_repo: TickerRepository,
        signal_repo: SignalRepository,
        alert_repo: AlertRepository,
        suggestion_repo: SuggestionRepository,
        sentiment_repo: SentimentRepository,
        earnings_repo: EarningsRepository,
        transaction_repo: TransactionRepository,
        config_repo: ConfigRepository,
        inference_pipeline: InferencePipeline,
        alert_evaluator: AlertEvaluator,
        alert_deduplicator: AlertDeduplicator,
        suggestion_engine: SuggestionEngine,
        position_deriver,
        session,
    ) -> None:
        self._ticker_repo = ticker_repo
        self._signal_repo = signal_repo
        self._alert_repo = alert_repo
        self._suggestion_repo = suggestion_repo
        self._sentiment_repo = sentiment_repo
        self._earnings_repo = earnings_repo
        self._transaction_repo = transaction_repo
        self._config_repo = config_repo
        self._inference_pipeline = inference_pipeline
        self._alert_evaluator = alert_evaluator
        self._alert_deduplicator = alert_deduplicator
        self._suggestion_engine = suggestion_engine
        self._position_deriver = position_deriver
        self._session = session

    def run_for_all(self) -> list[InferenceOrchestratorResult]:
        """
        Run inference for every ticker on the watchlist.

        Returns:
            List of InferenceOrchestratorResult, one per ticker.
        """
        tickers = self._ticker_repo.get_watchlist()
        return [self.run_for_ticker(t.symbol) for t in tickers]

    def run_for_ticker(self, symbol: str) -> InferenceOrchestratorResult:
        """
        Run inference for one ticker and handle all downstream work.

        Args:
            symbol: The ticker to run inference for.

        Returns:
            InferenceOrchestratorResult. Never raises.
        """
        try:
            # Step 1 — run inference
            prediction = self._inference_pipeline.predict(symbol)

            # Step 2 — persist signal
            signal_row = SignalModel(
                ticker=symbol,
                value=prediction.signal.value,
                probability=prediction.probability,
                model_name=prediction.model_name,
                fwd_days=prediction.fwd_days,
                created_at=prediction.predicted_at,
            )
            self._signal_repo.add(signal_row)
            self._session.flush()  # Get the auto-generated id before continuing

            # Step 3 — gather inputs for alert evaluation
            previous_signal_row = self._signal_repo.get_history(symbol, limit=2)
            previous_signal = None
            if len(previous_signal_row) >= 2:
                previous_signal = SignalValue(previous_signal_row[1].value)

            position = self._position_deriver.derive(symbol)
            current_prices = self._get_current_price(symbol)

            upcoming_earnings = set()
            upcoming = self._earnings_repo.get_upcoming_for_ticker(symbol)
            if upcoming:
                upcoming_earnings.add(symbol)

            alert_config = self._config_repo.get_all_for_category(
                ConfigCategory.ALERTS
            )

            evaluator_inputs = AlertEvaluatorInputs(
                current_signals={symbol: prediction.signal},
                previous_signals={symbol: previous_signal} if previous_signal else {},
                signal_ids={symbol: signal_row.id},
                positions={symbol: position},
                current_prices={symbol: current_prices},
                upcoming_earnings=upcoming_earnings,
                expiring_option_ids=[],
                expiring_option_symbols={},
                concentration_warnings=[],
                correlation_warnings=[],
                config=alert_config,
            )

            # Step 4 — evaluate and deduplicate alerts
            candidates = self._alert_evaluator.evaluate(evaluator_inputs)
            unique_candidates = self._alert_deduplicator.filter(
                candidates, alert_config
            )

            # Step 5 — persist unique alerts
            for candidate in unique_candidates:
                alert_row = AlertModel(
                    ticker=candidate.symbol,
                    triggering_signal_id=candidate.triggering_signal_id,
                    alert_type=candidate.alert_type.value,
                    severity=candidate.severity.value,
                    message=candidate.message,
                )
                self._alert_repo.add(alert_row)

            # Step 6 — generate suggestion
            sentiment_records = self._sentiment_repo.get_by_ticker(symbol)
            weighted_sentiment = 0.0
            if sentiment_records:
                total_weight = sum(r.source_weight for r in sentiment_records)
                if total_weight > 0:
                    weighted_sentiment = sum(
                        r.raw_score * r.source_weight for r in sentiment_records
                    ) / total_weight

            suggestion_inputs = SuggestionInputs(
                symbol=symbol,
                signal=prediction.signal,
                probability=prediction.probability,
                filter_result=prediction.filter_result,
                position=position,
                weighted_sentiment=weighted_sentiment,
                sentiment_record_count=len(sentiment_records),
                has_upcoming_earnings=bool(upcoming_earnings),
                earnings_report_date=upcoming.report_date if upcoming else None,
                last_signal_at=prediction.predicted_at,
                last_suggestion_at=self._suggestion_repo.get_latest_timestamp(symbol),
                last_sentiment_at=self._sentiment_repo.get_latest_timestamp(symbol),
                last_transaction_at=self._transaction_repo.get_latest_timestamp(symbol),
            )

            suggestion = self._suggestion_engine.generate(suggestion_inputs)
            suggestion_created = False

            if suggestion is not None:
                suggestion_row = SuggestionLogModel(
                    ticker=symbol,
                    signal_id=signal_row.id,
                    recommendation=suggestion.recommendation,
                    explanation=suggestion.explanation,
                    position_direction=suggestion.position_direction.value,
                    position_intent=suggestion.position_intent.value,
                    sentiment_summary=suggestion.sentiment_summary,
                    earnings_context=suggestion.earnings_context,
                    created_at=suggestion.generated_at,
                )
                self._suggestion_repo.add(suggestion_row)
                suggestion_created = True

            self._session.commit()

            return InferenceOrchestratorResult(
                symbol=symbol,
                success=True,
                prediction=prediction,
                alerts_created=len(unique_candidates),
                suggestion_created=suggestion_created,
            )

        except StockPredictorError as exc:
            self._session.rollback()
            return InferenceOrchestratorResult(
                symbol=symbol,
                success=False,
                prediction=None,
                alerts_created=0,
                suggestion_created=False,
                error=exc,
            )

    def _get_current_price(self, symbol: str) -> float:
        """Get the most recent close price for a ticker from MarketCache."""
        try:
            from src.data.market_cache import MarketCache
            # MarketCache is injected via the inference pipeline's data_loader
            # For now read directly — refactor to inject cache if needed
            cache = self._inference_pipeline._data_loader._cache
            df = cache.read(symbol)
            return float(df["close"].iloc[-1])
        except Exception:
            return 0.0