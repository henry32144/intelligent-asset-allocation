'''
Write test functions in this file
'''
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from flask import Blueprint, abort, request, render_template, jsonify
from database.tables.user import User
from database.database import db
from database.tables.crawling_data import CrawlingData, read_csv, daily_update
from database.tables.price import StockPrice, save_history_stock_price_to_db, update_daily_stock_price
from database.tables.company import Company, crawl_sp500_info, save_company
from database.tables.volatility import Volatility

from database.tables.output_news import OutputNews, news_to_json
from database.tables.portfolio import Portfolio
from model.get_news_keysent import KeysentGetter, test_url

from model.predict_Q import predict_Q
from model.markowitz import Markowitz
from model.equal_weight import EqualWeight
from model.black_litterman import Black_Litterman
from model.sp500 import SP500
from model.predict_Q_NLP import predict_Q_NLP

test_cases = Blueprint('test_cases', __name__)

@test_cases.route('/query_user_test')
def query():
    result = User.query.all()
    response = 'Not found'
    if result != None:
        response = result[0].username
    return response

@test_cases.route('/create_user_test')
def create_user_test():
    result = User.query.filter_by(username='test').first()
    print(result)
    if result is None:
        # create test row
        test = User(username='Test User', email='test@example.com')
        db.session.add(test)
        db.session.commit()
        print("create new user")
    return ''

@test_cases.route('/query_crawling_test')
def crawling_test():
    result = CrawlingData.query.all()
    response = 'Not found'
    if result != None:
        response = result[0].news_title
    return response


@test_cases.route('/create_crawling_test')
def create_crawling_test():
    read_csv('./database/tables/test.csv')
    return ''

@test_cases.route('/daily_update')
def update():
    daily_update()
    return ''


@test_cases.route('/create_price_db')
def create_pricetable_test():
    save_history_stock_price_to_db()
    return ''

@test_cases.route('/daily_update_stockprice')
def daily_update_stockprice():
    update_daily_stock_price('./database/tables/sp500tickers.pkl')
    return ''

@test_cases.route('/save_company')
def save_company_to_db():
    save_company()
    return ''


@test_cases.route('/outputnews')
def json_test():
    _ = news_to_json('Alphabet Inc.')
    print("to json")
    res = {}
    res["news"] = _
    return ''

@test_cases.route('/get_news')
def get_news():
    # _ = test_url()
    getter = KeysentGetter()
    print("create getter")
    getter.url2news()
    print("url 2 news")
    # getter.get_news()
    # getter.to_db()
    # print("to db")
    return ''


@test_cases.route('/get_stock_price')
def get_predicted_Q():
    with open('./model/sp500tickers.pkl', 'rb') as f:
        tickers = pickle.load(f)

    Q_predict_dict = defaultdict(list)
    real_price_dict = defaultdict(list)

    for tick in tqdm(tickers):
        predicted_price, real_stock_price = predict_Q(tick)
        Q_predict_dict[tick].append(predicted_price)
        real_price_dict[tick].append(real_stock_price)

    with open('predict_dict.pkl', 'wb') as f:
        pickle.dump(Q_predict_dict, f)

    with open('real_price_dict.pkl', 'wb') as f:
        pickle.dump(real_price_dict, f)

    return ''

@test_cases.route('/test')
def test():
    selected_tickers = ['GOOG', 'CVX', 'CB', 'CI', 'AAPL']
    marko = Markowitz(selected_tickers)
    all_weights = marko.get_all_weights()

    # all_values, all_return = marko.get_backtest_result()
    
    print('all_weights:', all_weights)
    # print('all_values:', all_values)
    # print('all_return:', all_return)

    # matplotlib.use('agg')
    # plt.plot(all_values, label='Mean-Var Portfolio')
    # plt.show()
    # plt.savefig('markowitz.png')

    date, all_values = marko.get_backtest_result()

    test_dict = {
        "data": {"all_weights": all_weights,
        "all_values": all_values,
        "date": date.tolist()}
    }

    return jsonify(test_dict)

@test_cases.route('/test_sp500')
def test_sp500():
    sp_500 = SP500()
    sp500_values = sp_500.get_backtest_result()

    return ''


@test_cases.route('/test_BL')
def test_black_litterman():
    selected_tickers = ['GOOG', 'CVX', 'CB', 'CI', 'AAPL']
    BL = Black_Litterman(selected_tickers)
    all_weights = BL.get_all_weights()
    date, all_values = BL.get_backtest_result()
    print(all_values)

    return ''

@test_cases.route("/find_error_bl", methods=['GET'])
def get_all_company():
    json_data = request.get_json()
    result = Company.query.all()
    basic_company = ["AMZN"]
    error_list = []
    # Response template

    if len(result) > 0:
        for r in tqdm(result):
            try:
                selected_tickers = basic_company + [r.symbol]
                BL = Black_Litterman(selected_tickers)
                all_weights = BL.get_all_weights()
                date, all_values = BL.get_backtest_result()
            except Exception as e:
                print(str(e))
                error_list.append(r.symbol)
    return jsonify(error_list)

@test_cases.route('/test_equal')
def test_equal_weight():
    selected_tickers = ['GOOG', 'CVX', 'CB', 'CI', 'AAPL']
    ew = EqualWeight(selected_tickers)
    
    date, all_values = ew.get_backtest_result()

    return ''


@test_cases.route('/predict_Q_NLP')
def test_predict_nlp():
    predicted_price, real_stock_price, test_df = predict_Q_NLP('GOOG')

    # print(train_df.head().to_string())
    print(test_df.head().to_string())


    return ''