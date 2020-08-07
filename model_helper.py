from flask import Blueprint, abort, request, render_template

model_helper = Blueprint('model_helper', __name__)

def interval_test():
    print("interval_test")

def cron_test():
    print("cron_test")