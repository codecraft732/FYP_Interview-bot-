"""Main Flask entry point for the AI-Based Self-Assessment Interview Bot."""
import os

from flask import Flask, render_template

from routes.api import api_bp, init_services

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(FRONTEND_DIR, "templates"),
        static_folder=os.path.join(FRONTEND_DIR, "static"),
    )
    app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max video upload
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    init_services(BASE_DIR)
    app.register_blueprint(api_bp)

    @app.after_request
    def disable_cache(response):
        """Prevent browser from serving stale cached HTML/CSS/JS during local runs."""
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="127.0.0.1", port=5000, debug=False)

app = create_app()
