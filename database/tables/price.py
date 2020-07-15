from main import db

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


