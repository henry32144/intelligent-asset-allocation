from database.database import db

class ModelOutput(db.Model):
	__tablename__ = "crawling_data"
	id = db.Column(Integer, primary_key=True)
	user_id = db.Column(db.Integer)
	date = db.Column(db.Date)
	model_output = db.Column(db.String)

	def __init__(self, user_id, model_output, date):
		
		self.user_id = user_id
		self.model_output = model_output
		self.date = date

	def __repr__(self):
		return "Model Output ('{}', '{}', '{}')".format(self.user_id, self.model_output, self.date)

