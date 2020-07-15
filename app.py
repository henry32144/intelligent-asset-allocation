from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from database.database import db
from test_cases import test_cases


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/database.db'
app.register_blueprint(test_cases)

@app.route('/')
def main():
    return ''

if __name__ == "__main__":
    db.app = app
    db.init_app(app)
    db.create_all()
    # admin = User(username='admin', email='admin@example.com')
    # db.session.merge(admin)
    # db.session.commit()
    app.run(debug=True)