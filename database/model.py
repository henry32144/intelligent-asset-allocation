from main import db

class Table(db.Model):
	index = db.Column(db.Integer, primary_key = True)
	content = db.Column(db.String(80))
	def __init__(self. content):
		self.content = content

	def __repr__(self):
		return '<Table %r>' %self.content
	