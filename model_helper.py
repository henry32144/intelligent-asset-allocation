from flask import Blueprint, abort, request, render_template

model_helper = Blueprint('model_helper', __name__)

@model_helper.route('/')
def show():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        abort(404)

def interval_test():
    print("interval_test")

def cron_test():
    print("cron_test")