from flask import Blueprint, abort, request, render_template

model_helper = Blueprint('model_helper', __name__)

@model_helper.route('/')
def show():
    try:
        print("123")
        return render_template("test.html")
    except TemplateNotFound:
        abort(404)

def interval_test():
    print("interval_test")

def cron_test():
    print("cron_test")