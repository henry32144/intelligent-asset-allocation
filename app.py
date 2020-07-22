from flask import Flask
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_apscheduler import APScheduler

from config import Config
from front_end import front_end
from model_helper import model_helper
from initialize import initialize
from database.database import db
from test_cases import test_cases

app = Flask(__name__)
app.config.from_object(Config())
scheduler = APScheduler()

app.register_blueprint(front_end)
app.register_blueprint(test_cases)
app.register_blueprint(model_helper)

if __name__ == "__main__":
    db.app = app
    db.init_app(app)
    db.create_all()
    initialize()
    app.run(debug=True)