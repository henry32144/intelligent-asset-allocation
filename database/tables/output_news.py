from database.database import db
import json
from flask import jsonify
from sqlalchemy import Column, Integer, String, DateTime, Date
from pprint import pprint

class OutputNews(db.Model):
	__tablename__ = "output_news"
	id = db.Column(Integer, primary_key=True)
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
	_dict = {
		"id": 0,
		"title": 0,
		"date": "",
		"paragraph":'',
		"keysent": [],
		"company": ""
	}
	# data = OutputNews.query.filter_by(company = company, date = datetime.now().date()- timedelta(days=1))
	data = OutputNews.query.filter(OutputNews.company.like(company)).all()
	print('get data')
	for r in data:
		a = _dict
		k = r.keysent.split('%%')
		a['keysent'] = k
		para = r.paragraph.split('%%')
		for p in para:
			p.replace('','')
		# print(para)
		a["id"] = r.id
		a["date"] = r.date
		a["company"] = r.company
		a['paragraph'] = para
		a['title'] = r.news_title
		result.append(a)

	# pprint(result[0])
	# return json.dumps(result) 
	return result