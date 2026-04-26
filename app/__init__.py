# app/__init__.py
#
# Flask application factory.
#
# Calling create_app() returns a fully configured Flask + SocketIO pair.
# Blueprints are registered here; WebSocket event handlers are defined here
# because they need direct access to the SocketIO instance.
#
# Why a factory instead of a module-level app object?
# - Makes the app testable (each test can call create_app() fresh)
# - Avoids circular import issues as the codebase grows
# - Keeps configuration explicit and in one place

from datetime import datetime

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# SocketIO is created at module level so blueprints and the factory
# can both reference it without circular imports.
socketio = SocketIO()

# Stores chat history per connected WebSocket session.
# Keyed by Flask-SocketIO session id (request.sid).
_conversation_histories: dict = {}


def create_app() -> tuple[Flask, SocketIO]:
    """
    Create and configure the Flask application.

    Returns (app, socketio) so run.py can call socketio.run(app, ...).
    """
    app = Flask(
        __name__,
        # Templates and static files are one level up from app/
        template_folder="../templates",
        static_folder="../static",
    )
    app.config["SECRET_KEY"] = "stock_predictor_secret"

    # ── Register blueprints ────────────────────────────────────────────────
    from app.routes import api_bp, portfolio_bp, options_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(options_bp)

    # ── Index page (no blueprint — single route, lives here) ───────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Initialize SocketIO against this app instance ──────────────────────
    socketio.init_app(app, cors_allowed_origins="*")

    # ── WebSocket event handlers ───────────────────────────────────────────
    # These are defined inside create_app so they close over `socketio`
    # without needing a global import.

    @socketio.on("connect")
    def handle_connect():
        _conversation_histories[request.sid] = []
        emit("connected", {"message": "Connected to Stock Prediction Bot"})

    @socketio.on("disconnect")
    def handle_disconnect():
        _conversation_histories.pop(request.sid, None)

    @socketio.on("message")
    def handle_message(data):
        sid = request.sid
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return

        if sid not in _conversation_histories:
            _conversation_histories[sid] = []

        history = _conversation_histories[sid]
        emit("thinking", {"status": "thinking"})

        try:
            from app.services.bot import chat
            response, updated_history = chat(user_msg, history)
            _conversation_histories[sid] = updated_history
            emit("response", {
                "message": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
        except Exception as e:
            emit("response", {
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

    return app, socketio
