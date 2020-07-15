from database.database import db
import pandas as pd

class StockPrice(db.Model):
    __tablename__ = 'price'

    id_ = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DataTime)
    high = db.Column(db.Float(precision=4))
    low = db.Column(db.Float(precision=4))
    open_ = db.Column(db.Float(precision=4))
    close = db.Column(db.Float(precision=4))
    vol = db.Column(db.Integer)
    adj_close = db.Column(db.Float(precision=4))
    query = db.Column(db.String(20))


    def __init__(self, date, high, low, open_, close, vol, adj_close, query):
        self.date = date
        self.high = high
        self.low = low
        self.open = open_
        self.close = close
        self.vol = vol
        self.adj_close = adj_close
        self.query = query

    def __repr__(self):
        return "Stock Price ('{}', '{}', '{}', '{}')".format(self.date, self.vol, self.adj_close, self.query)


def save_to_db(filename):
	df = pd.read_csv(filename)
	_lst = []
	for index, row in df.iterrows():
		_lst.append(StockPrice(row['Date'], row['High'], row['Low'], row['Open'], row['Close'], row['Volume'], row['Adj Close'], row['query']))
	db.session.add_all(_lst)
