from flask import Blueprint, jsonify, abort, request, render_template, send_from_directory
from database.tables.user import User
from database.tables.portfolio import Portfolio
from database.tables.company import Company
from database.database import db
from database.tables.output_news import OutputNews, news_to_json
from model.markowitz import Markowitz
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
            response['userId'] = result.id
            response['userName'] = result.user_name
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

@front_end.route("/portfolio", methods=['POST'])
def get_user_portfolio():
    json_data = request.get_json()

    result = Portfolio.query.filter_by(user_id=json_data.get("userId")).all()
    response = {
        'data': [],
        'isSuccess': False,
        'errorMsg': ""
    }
    for r in result:
        response['data'].append(r.to_json())
        
    response['isSuccess'] = True
    print(response)
    return jsonify(response)

@front_end.route("/portfolio/create", methods=['POST'])
def create_portfolio():
    json_data = request.get_json()

    new_portfolio = Portfolio(
        user_id=json_data.get("userId"),
        portfolio_name=str(json_data.get("portfolioName")),
        portfolio_stocks=''
    )

    db.session.add(new_portfolio)
    db.session.commit()
    db.session.refresh(new_portfolio)

    print(new_portfolio.portfolio_name)

    response = {
        'isSuccess': True,
        'data': new_portfolio.to_json(),
        'errorMsg': ""
    }
    
    return jsonify(response)

@front_end.route("/portfolio/save", methods=['POST'])
def save_portfolio():
    json_data = request.get_json()
    print(json_data)
    response = {
        'isSuccess': False,
        'errorMsg': ""
    }
    try:
        portfolio_id = json_data.get("portfolioId")
        result = Portfolio.query.filter_by(id=portfolio_id).first()
        portfolio_stocks = json_data.get("portfolioStocks", [''])
        portfolio_stocks_str = "%%".join(portfolio_stocks)
        result.portfolio_stocks = portfolio_stocks_str
        print(result)
        db.session.commit()
        response['isSuccess'] = True
    except Exception as e:
        response['errorMsg'] = str(e)
    
    return jsonify(response)

@front_end.route("/news", methods=['POST'])
def get_news_by_names():
    json_data = request.get_json()
    print(json_data)
    response = {
        'isSuccess': False,
        'errorMsg': "",
        'data': []
    }
    company_names = json_data.get("companyNames", [])
    news = []
    for name in company_names:
        company_news = news_to_json(name)
        news += company_news
    print(news)
    response["isSuccess"] = True
    response["data"] = news
    
    return jsonify(response)

@front_end.route("/portfolio/test", methods=['POST'])
def get_model_predict():
    json_data = request.get_json()
    response = {
        'isSuccess': False,
        'errorMsg': "",
        'data': []
    }
    selected_tickers = json_data.get("stocks", [])

    if len(selected_tickers) == 0:
        response["isSuccess"] = False
        response["errorMsg"] = "Cannot find portfolio"
    else:
        marko = Markowitz(selected_tickers)
        all_weights = marko.get_all_weights()
        date, all_values = marko.get_backtest_result()

        time_interval = 1
        for i in range(len(all_weights)):
            all_weights[i] = all_weights[i][::time_interval]
        
        response["data"] = {
            "all_weights": all_weights,
            "all_values": all_values[::time_interval],
            "date": date.tolist()[::time_interval]
        }
        response["isSuccess"] = True

    return jsonify(response)

# @front_end.route("/portfolio/calculate", methods=['POST'])
# def get_news_by_names():
#     json_data = request.get_json()
#     print(json_data)
#     response = {
#         'isSuccess': False,
#         'errorMsg': "",
#         'data': []
#     }
#     portfolio_id = json_data.get("portfolioId", None)
#     model_type = json_data.get("modelType", "basic")

#     if portfolio_id is None:
#         response["isSuccess"] = False
#         response["errorMsg"] = "Cannot find portfolio"
#     else:
#         pass

#     response["isSuccess"] = True
#     response["data"] = ""
#     return jsonify(response)