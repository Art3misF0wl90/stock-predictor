# app/routes/__intit__.py
# Exports all blueprints so app/__intit__.py can
# Register them in one import

from .api import api_bp
from .portfolio import portfolio_bp
from .options import options_bp