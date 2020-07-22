'''
Write test functions in this file
'''
from flask import Blueprint, abort, request, render_template
from database.tables.user import User
from database.database import db
from database.tables.crawling_data import CrawlingData, read_csv, daily_update
from database.tables.price import StockPrice, save_history_stock_price_to_db, update_daily_stock_price
from database.tables.company import Company, crawl_sp500_info, save_company
from database.tables.output_news import OutputNews, to_json
from model.get_news_keysent import KeysentGetter, test_url


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

@test_cases.route('/get_news')
def get_news():
    # _ = test_url()
    getter = KeysentGetter()
    print("create getter")
    getter.url2news()
    print("url 2 news")
    getter.to_db()
    print("to db")
    return ''
