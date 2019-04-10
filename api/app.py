from flask import Flask



def create_app() -> Flask:
    """Create a flask app instance."""

    flask_app = Flask('advance_api')

    # import blueprints
    from api.controller import advance_app
    flask_app.register_blueprint(advance_app)

    return flask_app
