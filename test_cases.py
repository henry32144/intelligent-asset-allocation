'''
Write test functions in this file
'''
from flask import Blueprint, abort, request, render_template
from database.tables.user import User
from database.database import db
from database.tables.crawling_data import CrawlingData, read_csv
from database.tables.price import StockPrice, save_stock_price_to_db

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


@test_cases.route('/create_pricetable_test')
def create_pricetable_test():
    save_stock_price_to_db('./database/tables/test_stock_price.csv')
    return ''