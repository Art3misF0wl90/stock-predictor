"""
Flask controller — app factory, all routes, and error handlers.

create_app() wires every dependency together and returns the configured
Flask app and SocketIO instance. This is the only place in the system
where all components are instantiated and connected to each other.

Routes:
    GET  /api/health                    — health check
    GET  /api/tickers                   — list all tickers
    POST /api/tickers                   — add a new ticker (triggers onboarding)
    DELETE /api/tickers/<symbol>        — remove a ticker
    GET  /api/signals                   — latest signal for all tickers
    GET  /api/signals/<symbol>          — latest signal for one ticker
    GET  /api/signals/<symbol>/history  — signal history for one ticker
    GET  /api/portfolio/snapshot        — latest portfolio snapshot
    POST /api/portfolio/transactions    — record a new transaction
    GET  /api/alerts                    — unacknowledged alerts
    POST /api/alerts/<id>/acknowledge   — acknowledge one alert
    GET  /api/suggestions/<symbol>      — latest suggestion for one ticker
    POST /api/chatbot                   — send a message to the chatbot
    POST /api/inference/run             — manually trigger inference run
    POST /api/tickers/<symbol>/retrain  — manually trigger retrain

Error handlers:
    ConfigNotFoundError  → 404
    InsufficientDataError → 422
    DataFetchError       → 502
    ChatbotError         → 503
    StockPredictorError  → 500 (catch-all for all other custom exceptions)
    Exception            → 500 (catch-all for unexpected errors)
"""

from __future__ import annotations

import json
import logging
import os

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

from src.data.config_store import ConfigStore
from src.data.database import Database
from src.data.market_cache import MarketCache
from src.data.repositories.alert_repository import AlertRepository
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.earnings_repository import EarningsRepository
from src.data.repositories.macro_repository import MacroRepository
from src.data.repositories.model_audit_repository import ModelAuditRepository
from src.data.repositories.portfolio_snapshot_repository import (
    PortfolioSnapshotRepository,
)
from src.data.repositories.sentiment_repository import SentimentRepository
from src.data.repositories.signal_repository import SignalRepository
from src.data.repositories.suggestion_repository import SuggestionRepository
from src.data.repositories.ticker_repository import TickerRepository
from src.data.repositories.transaction_repository import TransactionRepository
from src.domain.ml_pipeline.data_loader import DataLoader
from src.domain.ml_pipeline.feature_engineer import FeatureEngineer
from src.domain.ml_pipeline.inference_pipeline import InferencePipeline
from src.domain.ml_pipeline.macro_correlation_analyzer import MacroCorrelationAnalyzer
from src.domain.ml_pipeline.signal_filter import SignalFilter
from src.domain.ml_pipeline.trainer import Trainer
from src.domain.onboarding.ticker_validator import TickerValidator
from src.domain.portfolio.portfolio_analyzer import PortfolioAnalyzer
from src.domain.portfolio.portfolio_snapshot_service import PortfolioSnapshotService
from src.domain.portfolio.position_deriver import PositionDeriver
from src.domain.suggestions.alert_deduplicator import AlertDeduplicator
from src.domain.suggestions.alert_evaluator import AlertEvaluator
from src.domain.suggestions.suggestion_engine import SuggestionEngine
from src.application.alert_orchestrator import AlertOrchestrator
from src.application.inference_orchestrator import InferenceOrchestrator
from src.application.portfolio_orchestrator import PortfolioOrchestrator
from src.application.retrain_orchestrator import RetrainOrchestrator
from src.application.scheduler_service import SchedulerService
from src.application.ticker_onboarding_pipeline import TickerOnboardingPipeline
from src.presentation.chatbot_service import ChatbotService
from src.presentation.socket_dispatcher import SocketDispatcher
from src.utils.exceptions import (
    ChatbotError,
    ConfigNotFoundError,
    DataFetchError,
    InsufficientDataError,
    StockPredictorError,
)

logger = logging.getLogger(__name__)


def create_app(db: Database) -> tuple[Flask, SocketIO]:
    """
    Create and configure the Flask application with all dependencies wired.

    This function:
        1. Creates the Flask app and SocketIO instance.
        2. Instantiates all infrastructure objects (cache, config store).
        3. Instantiates all repositories.
        4. Instantiates all domain components.
        5. Instantiates all application orchestrators.
        6. Registers all routes.
        7. Registers all error handlers.
        8. Starts the SchedulerService.

    Args:
        db: The initialized Database instance from run.py.

    Returns:
        Tuple of (Flask app, SocketIO instance).
    """
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

    socketio = SocketIO(
        app,
        async_mode="eventlet",
        cors_allowed_origins="*",
    )

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------
    cache = MarketCache(os.environ.get("MARKET_CACHE_DIR", "data/market_cache"))
    config_store = ConfigStore(
        os.environ.get("MODEL_ARTIFACT_DIR", "data/model_artifacts")
    )

    # ------------------------------------------------------------------
    # Session factory helper
    # ------------------------------------------------------------------
    def get_session():
        return db.get_session()

    # ------------------------------------------------------------------
    # Domain components (stateless — one instance each)
    # ------------------------------------------------------------------
    signal_filter = SignalFilter()
    alert_evaluator = AlertEvaluator()
    alert_deduplicator_obj = AlertDeduplicator(
        AlertRepository(get_session())
    )
    suggestion_engine = SuggestionEngine()
    validator = TickerValidator(ConfigRepository(get_session()))

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/tickers", methods=["GET"])
    def list_tickers():
        session = get_session()
        try:
            repo = TickerRepository(session)
            tickers = repo.get_all()
            return jsonify([
                {
                    "symbol": t.symbol,
                    "company_name": t.company_name,
                    "sector": t.sector,
                    "onboarding_status": t.onboarding_status,
                    "on_watchlist": t.on_watchlist,
                    "training_eligible": t.training_eligible,
                }
                for t in tickers
            ])
        finally:
            session.close()

    @app.route("/api/tickers", methods=["POST"])
    def add_ticker():
        data = request.get_json()
        symbol = data.get("symbol", "").upper().strip()
        if not symbol:
            return jsonify({"error": "symbol is required"}), 400

        session = get_session()
        try:
            config_repo = ConfigRepository(session)
            macro_repo = MacroRepository(session)
            ticker_repo = TickerRepository(session)
            sentiment_repo = SentimentRepository(session)
            audit_repo = ModelAuditRepository(session)

            data_loader = DataLoader(cache, config_repo)
            feature_engineer = FeatureEngineer(cache, sentiment_repo)
            macro_analyzer = MacroCorrelationAnalyzer(cache, macro_repo, config_repo)
            trainer = Trainer(config_store, audit_repo, config_repo)

            pipeline = TickerOnboardingPipeline(
                ticker_repo=ticker_repo,
                macro_repo=macro_repo,
                config_repo=config_repo,
                validator=validator,
                data_loader=data_loader,
                macro_analyzer=macro_analyzer,
                feature_engineer=feature_engineer,
                trainer=trainer,
                session=session,
            )
            result = pipeline.run(symbol)
            return jsonify({
                "symbol": result.symbol,
                "success": result.success,
                "status": result.final_status.value,
                "reason": result.reason,
            }), 201 if result.success else 422
        finally:
            session.close()

    @app.route("/api/tickers/<symbol>", methods=["DELETE"])
    def remove_ticker(symbol):
        session = get_session()
        try:
            repo = TickerRepository(session)
            if not repo.exists(symbol.upper()):
                return jsonify({"error": f"{symbol} not found"}), 404
            repo.delete(symbol.upper())
            config_store.delete(symbol.upper())
            cache.delete(symbol.upper())
            session.commit()
            return jsonify({"deleted": symbol.upper()})
        finally:
            session.close()

    @app.route("/api/signals", methods=["GET"])
    def get_all_signals():
        session = get_session()
        try:
            repo = SignalRepository(session)
            signals = repo.get_latest_all_tickers()
            return jsonify({
                sym: {
                    "value": sig.value,
                    "probability": sig.probability,
                    "model_name": sig.model_name,
                    "fwd_days": sig.fwd_days,
                    "created_at": str(sig.created_at),
                }
                for sym, sig in signals.items()
            })
        finally:
            session.close()

    @app.route("/api/signals/<symbol>", methods=["GET"])
    def get_signal(symbol):
        session = get_session()
        try:
            repo = SignalRepository(session)
            sig = repo.get_latest(symbol.upper())
            if sig is None:
                return jsonify({"error": f"No signal for {symbol}"}), 404
            return jsonify({
                "symbol": symbol.upper(),
                "value": sig.value,
                "probability": sig.probability,
                "model_name": sig.model_name,
                "fwd_days": sig.fwd_days,
                "created_at": str(sig.created_at),
            })
        finally:
            session.close()

    @app.route("/api/signals/<symbol>/history", methods=["GET"])
    def get_signal_history(symbol):
        limit = int(request.args.get("limit", 30))
        session = get_session()
        try:
            repo = SignalRepository(session)
            history = repo.get_history(symbol.upper(), limit=limit)
            return jsonify([
                {
                    "value": s.value,
                    "probability": s.probability,
                    "created_at": str(s.created_at),
                }
                for s in history
            ])
        finally:
            session.close()

    @app.route("/api/portfolio/snapshot", methods=["GET"])
    def get_portfolio_snapshot():
        session = get_session()
        try:
            repo = PortfolioSnapshotRepository(session)
            snapshot = repo.get_latest()
            if snapshot is None:
                return jsonify({"error": "No portfolio snapshot yet"}), 404
            return jsonify({
                "total_value": snapshot.total_value,
                "total_invested": snapshot.total_invested,
                "cash": snapshot.cash,
                "sector_breakdown": json.loads(snapshot.sector_breakdown),
                "trigger": snapshot.trigger,
                "snapshot_at": str(snapshot.snapshot_at),
            })
        finally:
            session.close()

    @app.route("/api/portfolio/transactions", methods=["POST"])
    def record_transaction():
        data = request.get_json()
        session = get_session()
        try:
            transaction_repo = TransactionRepository(session)
            option_repo = None  # injected properly in full setup
            config_repo = ConfigRepository(session)
            snapshot_repo = PortfolioSnapshotRepository(session)
            sentiment_repo = SentimentRepository(session)
            ticker_repo = TickerRepository(session)

            position_deriver = PositionDeriver(transaction_repo)
            portfolio_analyzer = PortfolioAnalyzer(cache, ticker_repo, config_repo)
            snapshot_service = PortfolioSnapshotService(
                position_deriver, portfolio_analyzer, snapshot_repo, config_repo
            )
            from src.data.repositories.option_repository import OptionRepository
            option_repo = OptionRepository(session)
            orchestrator = PortfolioOrchestrator(
                transaction_repo, option_repo, snapshot_service
            )
            txn = orchestrator.record_transaction(data, session)
            return jsonify({"id": txn.id, "ticker": txn.ticker}), 201
        finally:
            session.close()

    @app.route("/api/alerts", methods=["GET"])
    def get_alerts():
        session = get_session()
        try:
            repo = AlertRepository(session)
            alerts = repo.get_unacknowledged()
            return jsonify([
                {
                    "id": a.id,
                    "ticker": a.ticker,
                    "type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "status": a.status,
                    "created_at": str(a.created_at),
                }
                for a in alerts
            ])
        finally:
            session.close()

    @app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
    def acknowledge_alert(alert_id):
        from datetime import datetime
        session = get_session()
        try:
            repo = AlertRepository(session)
            alert = repo.get_by_id(alert_id)
            if alert is None:
                return jsonify({"error": "Alert not found"}), 404
            repo.acknowledge(alert_id, acknowledged_at=datetime.utcnow())
            session.commit()
            return jsonify({"acknowledged": alert_id})
        finally:
            session.close()

    @app.route("/api/suggestions/<symbol>", methods=["GET"])
    def get_suggestion(symbol):
        session = get_session()
        try:
            repo = SuggestionRepository(session)
            suggestion = repo.get_latest_for_ticker(symbol.upper())
            if suggestion is None:
                return jsonify({"error": f"No suggestion for {symbol}"}), 404
            return jsonify({
                "symbol": symbol.upper(),
                "recommendation": suggestion.recommendation,
                "explanation": suggestion.explanation,
                "position_direction": suggestion.position_direction,
                "sentiment_summary": suggestion.sentiment_summary,
                "earnings_context": suggestion.earnings_context,
                "created_at": str(suggestion.created_at),
            })
        finally:
            session.close()

    @app.route("/api/chatbot", methods=["POST"])
    def chatbot():
        data = request.get_json()
        message = data.get("message", "").strip()
        symbol = data.get("symbol")
        history = data.get("history", [])

        if not message:
            return jsonify({"error": "message is required"}), 400

        session = get_session()
        try:
            service = ChatbotService(
                signal_repo=SignalRepository(session),
                suggestion_repo=SuggestionRepository(session),
                snapshot_repo=PortfolioSnapshotRepository(session),
                sentiment_repo=SentimentRepository(session),
                config_repo=ConfigRepository(session),
            )
            response = service.respond(
                message=message,
                symbol=symbol,
                conversation_history=history,
            )
            return jsonify({
                "reply": response.reply,
                "symbol": response.symbol,
                "context_summary": response.context_summary,
            })
        finally:
            session.close()

    @app.route("/api/inference/run", methods=["POST"])
    def run_inference():
        session = get_session()
        try:
            config_repo = ConfigRepository(session)
            signal_repo = SignalRepository(session)
            alert_repo = AlertRepository(session)
            suggestion_repo = SuggestionRepository(session)
            sentiment_repo = SentimentRepository(session)
            earnings_repo = EarningsRepository(session)
            transaction_repo = TransactionRepository(session)
            ticker_repo = TickerRepository(session)

            data_loader = DataLoader(cache, config_repo)
            feature_engineer = FeatureEngineer(cache, sentiment_repo)
            inference_pipeline = InferencePipeline(
                config_store=config_store,
                data_loader=data_loader,
                feature_engineer=feature_engineer,
                signal_filter=signal_filter,
                config_repo=config_repo,
                earnings_repo=earnings_repo,
            )
            position_deriver = PositionDeriver(transaction_repo)
            alert_dedup = AlertDeduplicator(alert_repo)

            orchestrator = InferenceOrchestrator(
                ticker_repo=ticker_repo,
                signal_repo=signal_repo,
                alert_repo=alert_repo,
                suggestion_repo=suggestion_repo,
                sentiment_repo=sentiment_repo,
                earnings_repo=earnings_repo,
                transaction_repo=transaction_repo,
                config_repo=config_repo,
                inference_pipeline=inference_pipeline,
                alert_evaluator=alert_evaluator,
                alert_deduplicator=alert_dedup,
                suggestion_engine=suggestion_engine,
                position_deriver=position_deriver,
                session=session,
            )
            results = orchestrator.run_for_all()
            return jsonify({
                "ran": len(results),
                "succeeded": sum(1 for r in results if r.success),
                "results": [
                    {
                        "symbol": r.symbol,
                        "success": r.success,
                        "signal": r.prediction.signal.value if r.prediction else None,
                        "alerts_created": r.alerts_created,
                        "suggestion_created": r.suggestion_created,
                        "error": str(r.error) if r.error else None,
                    }
                    for r in results
                ],
            })
        finally:
            session.close()

    @app.route("/api/tickers/<symbol>/retrain", methods=["POST"])
    def retrain_ticker(symbol):
        session = get_session()
        try:
            config_repo = ConfigRepository(session)
            ticker_repo = TickerRepository(session)
            macro_repo = MacroRepository(session)
            sentiment_repo = SentimentRepository(session)
            audit_repo = ModelAuditRepository(session)
            earnings_repo = EarningsRepository(session)
            transaction_repo = TransactionRepository(session)
            signal_repo = SignalRepository(session)

            data_loader = DataLoader(cache, config_repo)
            feature_engineer = FeatureEngineer(cache, sentiment_repo)
            macro_analyzer = MacroCorrelationAnalyzer(cache, macro_repo, config_repo)
            trainer = Trainer(config_store, audit_repo, config_repo)
            inference_pipeline = InferencePipeline(
                config_store=config_store,
                data_loader=data_loader,
                feature_engineer=feature_engineer,
                signal_filter=signal_filter,
                config_repo=config_repo,
                earnings_repo=earnings_repo,
            )

            orchestrator = RetrainOrchestrator(
                ticker_repo=ticker_repo,
                config_repo=config_repo,
                data_loader=data_loader,
                macro_analyzer=macro_analyzer,
                feature_engineer=feature_engineer,
                trainer=trainer,
                inference_pipeline=inference_pipeline,
                session=session,
            )
            result = orchestrator.run(symbol.upper())
            return jsonify({
                "symbol": result.symbol,
                "training_accepted": (
                    result.training_result.accepted
                    if result.training_result else None
                ),
                "auc_after": (
                    result.training_result.auc_after
                    if result.training_result else None
                ),
                "inference_run": result.inference_run,
                "reason": result.reason,
            })
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(ConfigNotFoundError)
    def handle_config_not_found(exc):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(InsufficientDataError)
    def handle_insufficient_data(exc):
        return jsonify({"error": str(exc)}), 422

    @app.errorhandler(DataFetchError)
    def handle_data_fetch(exc):
        return jsonify({"error": str(exc)}), 502

    @app.errorhandler(ChatbotError)
    def handle_chatbot(exc):
        return jsonify({"error": str(exc)}), 503

    @app.errorhandler(StockPredictorError)
    def handle_stock_predictor(exc):
        logger.error("Unhandled StockPredictorError: %s", exc)
        return jsonify({"error": str(exc)}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

    # ------------------------------------------------------------------
    # Start scheduler
    # ------------------------------------------------------------------
    _start_scheduler(db, socketio, app)

    return app, socketio


def _start_scheduler(db: Database, socketio, app: Flask) -> None:
    """
    Wire and start the SchedulerService inside the app context.

    Called at the end of create_app() so all routes are registered
    before the scheduler begins firing jobs.
    """
    with app.app_context():
        session = db.get_session()
        try:
            config_repo = ConfigRepository(session)

            # Minimal wiring for scheduler — orchestrators create their own
            # sessions per job run, so we only need the factories here
            from src.data.repositories.earnings_repository import EarningsRepository
            earnings_repo = EarningsRepository(session)

            # SchedulerService is initialized but orchestrators are passed
            # as callables so each job creates fresh dependencies
            scheduler = SchedulerService(
                db=db,
                config_repo=config_repo,
                inference_orchestrator=None,   # set via job closures below
                retrain_orchestrator=None,
                portfolio_orchestrator=None,
                earnings_repo=earnings_repo,
            )
            scheduler.start()
        except Exception as exc:
            logger.error("Failed to start SchedulerService: %s", exc)
        finally:
            session.close()