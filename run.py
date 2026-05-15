"""
run.py — application entrypoint for stock_predictor_v2.

Startup sequence:
    1. Load .env file
    2. Initialize Database (engine + session factory)
    3. Run migrations (create tables if missing)
    4. Seed SYSTEM_CONFIG with defaults if empty
    5. Call create_app() to wire all dependencies and register routes
    6. Start Flask + SocketIO

Run with:
    python run.py
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

# Load .env before any src imports so environment variables are available
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.data.database import Database
from src.data.repositories.config_repository import ConfigRepository
from src.presentation.flask_controller import create_app


def seed_config(db: Database) -> None:
    """
    Seed SYSTEM_CONFIG with default values on first startup.

    Reads defaults from config/default_config.json and inserts any
    keys that don't already exist in the database.

    Args:
        db: Initialized Database instance.
    """
    config_path = "config/default_config.json"
    if not os.path.exists(config_path):
        logger.warning("No default_config.json found — skipping config seed.")
        return

    with open(config_path) as f:
        entries = json.load(f)

    session = db.get_session()
    try:
        # Seed category rows first
        from src.data.database import SystemConfigCategoryModel
        from src.utils.types import ConfigCategory
        for category in ConfigCategory:
            existing = session.get(SystemConfigCategoryModel, category.value)
            if existing is None:
                session.add(SystemConfigCategoryModel(
                    category=category.value,
                    version=1,
                    description=f"{category.value} configuration",
                ))
        session.flush()

        repo = ConfigRepository(session)
        repo.seed(entries)
        session.commit()
        logger.info("SYSTEM_CONFIG seeded with %d entries.", len(entries))
    except Exception as exc:
        logger.error("Config seeding failed: %s", exc)
        session.rollback()
    finally:
        session.close()


def main() -> None:
    """Initialize the database, seed config, and start the server."""
    db_path = os.environ.get("DB_PATH", "data/stock_predictor.db")

    logger.info("Initializing database at %s", db_path)
    db = Database.init(db_path)
    db.migrate()
    logger.info("Database migration complete.")

    seed_config(db)

    logger.info("Creating Flask app and wiring dependencies.")
    app, socketio = create_app(db)

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    logger.info("Starting server on %s:%d", host, port)

    socketio.run(app, host=host, port=port, debug=False)


if __name__ == "__main__":
    main()