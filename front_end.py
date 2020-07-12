from flask import Blueprint, abort, request, render_template, send_from_directory

front_end = Blueprint('front_end', __name__)

@front_end.route("/manifest.json")
def manifest():
    return send_from_directory('./static', 'manifest.json')