from flask import Flask
from config import Config
from front_end import front_end
from model_helper import model_helper
from flask_apscheduler import APScheduler


app = Flask(__name__)
app.config.from_object(Config())
scheduler = APScheduler()

app.register_blueprint(front_end)
app.register_blueprint(model_helper)

if __name__ == "__main__":
    #scheduler = APScheduler()
    #scheduler.init_app(app)
    #scheduler.start()

    app.run(debug=True)