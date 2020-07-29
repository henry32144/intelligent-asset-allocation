from database.database import db


class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    portfolio_name = db.Column(db.String, nullable=False)
    portfolio_stocks = db.Column(db.String, nullable=False)

    def __init__(self, **kwargs):
        super(Portfolio, self).__init__(**kwargs)

    def __repr__(self):
        return '<Portfolio %r>' % self.portfolio_name
    
    def to_json(self):
        portfolio_stocks = self.portfolio_stocks.split('%%')
        if len(portfolio_stocks[0]) == 0:
            portfolio_stocks = []
        return {
                'id': self.id,
                'user_id': self.user_id,
                'portfolio_name': self.portfolio_name,
                'portfolio_stocks': portfolio_stocks,
                }