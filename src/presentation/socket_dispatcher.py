"""
SocketDispatcher — emits WebSocket events to connected dashboard clients.

Wraps Flask-SocketIO's emit() so the rest of the system never imports
flask_socketio directly. All WebSocket output goes through this class.

Events emitted:
    "alert"         — a new alert was created
    "signal_update" — a new signal was generated for a ticker
    "inference_done"— an inference run completed for all tickers
    "error"         — a server-side error the client should display

Depends on:
    flask_socketio.SocketIO — the SocketIO instance from Flask factory

Exposes:
    emit_alert(alert_dict)
    emit_signal_update(symbol, signal, probability)
    emit_inference_done(results)
    emit_error(message)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SocketDispatcher:
    """
    Emits WebSocket events to connected clients.

    One instance is created in create_app() and injected into
    AlertOrchestrator and InferenceOrchestrator.

    Usage:
        dispatcher = SocketDispatcher(socketio)
        dispatcher.emit_alert({...})
    """

    def __init__(self, socketio) -> None:
        """
        Args:
            socketio: The Flask-SocketIO instance from create_app().
        """
        self._socketio = socketio

    def emit_alert(self, alert_dict: dict) -> None:
        """
        Push a new alert to all connected dashboard clients.

        Args:
            alert_dict: Dict with keys: id, ticker, type, severity,
                        message, created_at.
        """
        try:
            self._socketio.emit("alert", alert_dict)
        except Exception as exc:
            logger.error("SocketDispatcher.emit_alert failed: %s", exc)
            raise

    def emit_signal_update(
        self,
        symbol: str,
        signal: str,
        probability: float,
    ) -> None:
        """
        Push a signal update for one ticker to all connected clients.

        Args:
            symbol: The ticker symbol.
            signal: Signal value string — "BUY", "HOLD", or "SELL".
            probability: Model confidence (0.0 to 1.0).
        """
        try:
            self._socketio.emit("signal_update", {
                "symbol": symbol,
                "signal": signal,
                "probability": round(probability, 4),
            })
        except Exception as exc:
            logger.error("SocketDispatcher.emit_signal_update failed: %s", exc)

    def emit_inference_done(self, results: list) -> None:
        """
        Push a summary of a completed inference run to all clients.

        Args:
            results: List of InferenceOrchestratorResult objects.
        """
        try:
            summary = [
                {
                    "symbol": r.symbol,
                    "success": r.success,
                    "signal": r.prediction.signal.value if r.prediction else None,
                    "alerts_created": r.alerts_created,
                    "suggestion_created": r.suggestion_created,
                }
                for r in results
            ]
            self._socketio.emit("inference_done", {"results": summary})
        except Exception as exc:
            logger.error("SocketDispatcher.emit_inference_done failed: %s", exc)

    def emit_error(self, message: str) -> None:
        """
        Push an error message to all connected clients.

        Args:
            message: Human-readable error description.
        """
        try:
            self._socketio.emit("error", {"message": message})
        except Exception as exc:
            logger.error("SocketDispatcher.emit_error failed: %s", exc)