from flask import Blueprint, jsonify, abort, request, render_template, send_from_directory

front_end = Blueprint('front_end', __name__)

@front_end.route("/manifest.json")
def manifest():
    return send_from_directory('./static', 'manifest.json')

@front_end.route("/post-test", methods=['POST'])
def post_test():
    json_data = request.get_json()
    test_res = {
        'data': json_data.get("selectedStocks", [])
    }
    return jsonify(test_res)