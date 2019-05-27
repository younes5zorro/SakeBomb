from flask import Flask



def create_app() -> Flask:
    """Create a flask app instance."""

    flask_app = Flask('advance_api')
    flask_app.config['JSON_SORT_KEYS'] = False
    # import blueprints
    from api.connection import advance_app
    from api.algos import advance_alogs
    from api.joins import advance_join
    from api.twitter import advance_tweet
    from api.static import advance_static


    flask_app.register_blueprint(advance_app)
    flask_app.register_blueprint(advance_join)
    flask_app.register_blueprint(advance_alogs)
    flask_app.register_blueprint(advance_tweet)
    flask_app.register_blueprint(advance_static)

    return flask_app
