from database.database import db
import json
from sqlalchemy import Column, Integer, String, DateTime, Date

class OutputNews(db.Model):
	__tablename__ = "output_news"
	id = db.Column(Integer, primary_key=True)
	news_title = db.Column(db.String)
	date = db.Column(db.Date)
	company = db.Column(db.String)
	paragraph = db.Column(db.String)
	keysent = db.Column(db.String)

	def __init__(self, news_title, date, company, url):
		self.news_title = news_title
		self.date = date.date()
		self.company = company
		self.paragraph = paragraph
		self.keysent = keysent

def to_json(company):
	result = []
	_dict = {
	"title": 0,
	"pargraph":0,
	"keysent":0
	}
	# data = OutputNews.query.filter_by(company = company, date = datetime.now().date()- timedelta(days=1))
	data = OutputNews.query.filter_by(company = company)
	for r in data:
		a = _dict
		a['title'] = r.news_title
		a['paragraph'] = r.paragraph.split("%")
		a['keysent'] = r.keysent.split('%')
		result.append(a)
	return json.dumps(result) 