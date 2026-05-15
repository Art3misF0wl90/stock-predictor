"""
RetrainOrchestrator — retrains one ticker's model and runs fresh inference.

Coordinates:
    1. DataLoader.fetch_incremental() — update market cache
    2. MacroCorrelationAnalyzer.analyze() — recompute macro relevance
    3. FeatureEngineer.build_train_matrix() — rebuild feature matrix
    4. Trainer.train() — train new model, accept or reject
    5. If accepted: trigger InferencePipeline.predict() immediately

run() never raises. All failures return RetrainResult with
training_result=None and a reason string.

Exposes:
    run(symbol) → RetrainResult
"""

from __future__ import annotations

from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.ticker_repository import TickerRepository
from src.domain.ml_pipeline.data_loader import DataLoader
from src.domain.ml_pipeline.feature_engineer import FeatureEngineer
from src.domain.ml_pipeline.inference_pipeline import InferencePipeline
from src.domain.ml_pipeline.macro_correlation_analyzer import MacroCorrelationAnalyzer
from src.domain.ml_pipeline.trainer import Trainer
from src.utils.exceptions import StockPredictorError
from src.utils.types import ConfigCategory, RetrainResult


class RetrainOrchestrator:
    """
    Retrains a ticker's model and optionally runs fresh inference.

    Usage:
        orchestrator = RetrainOrchestrator(...)
        result = orchestrator.run("AAPL")
    """

    def __init__(
        self,
        ticker_repo: TickerRepository,
        config_repo: ConfigRepository,
        data_loader: DataLoader,
        macro_analyzer: MacroCorrelationAnalyzer,
        feature_engineer: FeatureEngineer,
        trainer: Trainer,
        inference_pipeline: InferencePipeline,
        session,
    ) -> None:
        self._ticker_repo = ticker_repo
        self._config_repo = config_repo
        self._data_loader = data_loader
        self._macro_analyzer = macro_analyzer
        self._feature_engineer = feature_engineer
        self._trainer = trainer
        self._inference_pipeline = inference_pipeline
        self._session = session

    def run(self, symbol: str) -> RetrainResult:
        """
        Execute the full retrain sequence for one ticker.

        Args:
            symbol: The ticker to retrain.

        Returns:
            RetrainResult. Never raises.
        """
        try:
            # Step 1 — update market cache
            self._data_loader.fetch_incremental(symbol)

            # Step 2 — recompute macro relevance
            profile = self._macro_analyzer.analyze(symbol)
            self._session.commit()

            # Step 3 — rebuild feature matrix
            fwd_days = self._config_repo.get(
                ConfigCategory.TRAINING, "default_fwd_days"
            )
            feature_df, feat_cols, target = self._feature_engineer.build_train_matrix(
                symbol=symbol,
                macro_symbols=profile.selected_macros,
                fwd_days=fwd_days,
            )

            # Step 4 — train
            training_result = self._trainer.train(
                symbol=symbol,
                feature_df=feature_df,
                feat_cols=feat_cols,
                target=target,
                fwd_days=fwd_days,
            )
            self._session.commit()

            # Step 5 — if accepted, run fresh inference immediately
            inference_run = False
            if training_result.accepted:
                try:
                    self._inference_pipeline.predict(symbol)
                    inference_run = True
                except StockPredictorError:
                    # Inference failure after retrain is non-fatal —
                    # the new model is still active for the next cycle
                    pass

            return RetrainResult(
                symbol=symbol,
                training_result=training_result,
                inference_run=inference_run,
                reason=None,
            )

        except StockPredictorError as exc:
            return RetrainResult(
                symbol=symbol,
                training_result=None,
                inference_run=False,
                reason=str(exc),
            )