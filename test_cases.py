'''
Write test functions in this file
'''
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from flask import Blueprint, abort, request, render_template
from database.tables.user import User
from database.database import db
from database.tables.crawling_data import CrawlingData, read_csv, daily_update
from database.tables.price import StockPrice, save_history_stock_price_to_db, update_daily_stock_price
from database.tables.company import Company, crawl_sp500_info, save_company
from model.predict_Q import predict_Q

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


@test_cases.route('/create_pricetable_test')
def create_pricetable_test():
    save_history_stock_price_to_db('./database/tables/sp500tickers.pkl')
    return ''

@test_cases.route('/daily_update_stockprice')
def daily_update_stockprice():
    update_daily_stock_price('./database/tables/sp500tickers.pkl')
    return ''

@test_cases.route('/save_company')
def save_company_to_db():
    save_company()
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

    with open('predict_dict', 'wb') as f:
        pickle.dump(Q_predict_dict, f)

    with open('real_price_dict', 'wb') as f:
        pickle.dump(real_price_dict, f)

    return ''
