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

def to_json(company):
	result = []
	_dict = {
	"title": 0,
	"paragraph":'',
	"keysent":0
	}
	# data = OutputNews.query.filter_by(company = company, date = datetime.now().date()- timedelta(days=1))
	data = OutputNews.query.filter_by(company = company).all()
	print('get data')
	for r in data:
		a = _dict
		k = r.keysent.split('%%')
		a['keysent'] = k
		para = r.paragraph.split('%%')
		for p in para:
			p.replace('','')
		# print(para)
		a['paragraph'] = para
		a['title'] = r.news_title
		result.append(a)

	# pprint(result[0])
	# return json.dumps(result) 
	return jsonify(result)