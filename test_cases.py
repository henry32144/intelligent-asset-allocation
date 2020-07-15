'''
Write test functions in this file
'''
from flask import Blueprint, abort, request, render_template
from database.tables.user import User
from database.database import db
from database.tables.crawling_data import CrawlingData, read_csv
test_cases = Blueprint('test_cases', __name__)

@test_cases.route('/query_user_test')
def query():
    result = User.query.all()
    response = 'Not found'
    if len(result) > 0:
        response = result[0].user_name
    return response

@test_cases.route('/create_user_test')
def create_user_test():
    result = User.query.filter_by(user_name='test').first()
    print(result)
    if result is None:
        # create test row
        test = User(
            user_name='Test User',
            user_email='test@example.com',
            user_password='123'
        )
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