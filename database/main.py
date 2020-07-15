from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLALchemy
from datetime import datetime

app = Flask(__name__)

#MySql database
url = "mysql+pymysql://account:password@IP:3306/db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = url

db = SQLALchemy(app)
