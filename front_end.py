from flask import Blueprint, jsonify, abort, request, render_template, send_from_directory
from database.tables.user import User
from database.tables.company import Company
from database.database import db


front_end = Blueprint('front_end', __name__)

@front_end.route("/manifest.json")
def manifest():
    return send_from_directory('./static', 'manifest.json')

@front_end.route("/user/login", methods=['POST'])
def user_login():
    json_data = request.get_json()
    result = User.query.filter_by(user_email=json_data.get("userEmail")).first()
    # Response template
    response = {
        'isSuccess': False,
        'errorMsg': "Email or Password is incorrect"
    }
    if result is not None:
        # Not empty, check password
        if result.user_password == json_data.get("userPassword"):
            response['isSuccess'] = True
    else:
        response['errorMsg'] = "User not found"
    
    return jsonify(response)

@front_end.route("/user/signup", methods=['POST'])
def user_signup():
    json_data = request.get_json()
    result = User.query.filter_by(user_email=json_data.get("userEmail")).first()

    response = {
        'isSuccess': False,
        'errorMsg': "Signup failed"
    }

    if result is None:
        response['isSuccess'] = True
        new_user = User(
            user_name=json_data.get("userName"),
            user_email=json_data.get("userEmail"),
            user_password=json_data.get("userPassword")
        )
        db.session.add(new_user)
        db.session.commit()
    else:
        response['errorMsg'] = "User already exists"
    return jsonify(response)


@front_end.route("/company", methods=['GET'])
def get_all_company():
    json_data = request.get_json()
    result = Company.query.all()
    # Response template
    response = {
        'data': [],
        'isSuccess': False,
        'errorMsg': ""
    }
    if len(result) > 0:
        for r in result:
            response['data'].append(r.to_json())
        
        response['isSuccess'] = True
    else:
        response['errorMsg'] = "Get company failed"
    
    return jsonify(response)