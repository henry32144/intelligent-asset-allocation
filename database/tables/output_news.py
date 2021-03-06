from database.database import db
import json
from flask import jsonify
from sqlalchemy import Column, Integer, String, DateTime, Date, and_
from pprint import pprint
from datetime import datetime

class OutputNews(db.Model):
	__tablename__ = "output_news"
	id = db.Column(db.Integer, primary_key=True)
	news_title = db.Column(db.String)
	date = db.Column(db.Date)
	company = db.Column(db.String)
	paragraph = db.Column(db.String)
	keysent = db.Column(db.String)

	def __init__(self, news_title, date, company,paragraph, keysent):
		self.news_title = news_title
		self.date = date
		self.company = company
		self.paragraph = paragraph
		self.keysent = keysent

def news_to_json(company):
	result = []
	print(company)
	date1 = datetime( 2020,7,2 )
	date2 = datetime(2020,6,1)
	# data = OutputNews.query.filter(OutputNews.date <= date1).filter(OutputNews.date >= date2)
	#data = OutputNews.query.filter( and_( OutputNews.company == company, OutputNews.date > date2 ) ).all()
	data = OutputNews.query.filter( OutputNews.company == company ).all()
	print('get data')

	
	print(data)
	for r in data:
		k = r.keysent.split('%%')
		para = r.paragraph.split('%%')
		for p in para:
			p.replace('','')
		# print(para)
		a = {
			"id": r.id,
			"title": r.news_title,
			"date": r.date,
			"paragraph":para,
			"keysent": k,
			"company": r.company
		}
		result.append(a)
		print(r.news_title)
	return result



