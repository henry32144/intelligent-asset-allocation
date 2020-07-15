from database.database import db
import datetime
from sqlalchemy import Column, Integer, String, DateTime
import pandas as pd
from database.tables.daily_crawler import ArticleGetter

class CrawlingData(db.Model):
	__tablename__ = "crawling_data"
	id = db.Column(Integer, primary_key=True)
	news_title = db.Column(db.String)
	date = db.Column(db.DateTime)
	company = db.Column(db.String)
	url = db.Column(db.String)
	
	def __init__(self, news_title, date, company, url):
		self.news_title = news_title
		self.date = date
		self.company = company
		self.url = url

	def get_one_data(self, index):
		return 0

	def __repr__(self):
		return "Crawling data ('{}', '{}', '{}', '{}')".format(self.news_title, self.date, self.company, self.url)

def read_csv(filename):
	df = pd.read_csv(filename)
	_lst = []
	for index, row in df.iterrows():
		_lst.append(CrawlingData(row['title'], row['time'], row['query'], row['url']))
	db.session.add_all(_lst)
	db.session.commit()

def daily_update():
	base_url = 'https://www.reuters.com/search/news?blob='
	querys = ['Google']
	article_getter = ArticleGetter(base_url)
	for q in querys:
		df = article_getter.get_daily_news(q)
		_lst = []
		for index, row in df.iterrows():
			_lst.append(CrawlingData(news_title=row['title'], date=row['time'], company=q, url=row['url']))
		db.session.add_all(_lst)
		db.session.commit()