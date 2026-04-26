# run.py
#
# Entry point for the Stock Predictor web server.
#
# Usage:
#   python run.py
#
# The old app.py had its server startup inside `if __name__ == "__main__"`.
# That pattern doesn't work well once the app grows — importing app.py
# anywhere would trigger all the top-level code. This file is the single
# place that actually starts the server; everything else just defines things.

from app import create_app

app, socketio = create_app()

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  Stock Predictor Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("═" * 50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
