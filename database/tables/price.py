from database.database import db
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import pickle
from tqdm import tqdm

class StockPrice(db.Model):
    __tablename__ = 'price'

    id_ = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date)
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
        self.open_ = open_
        self.close = close
        self.vol = vol
        self.adj_close = adj_close
        self.query = query

    def __repr__(self):
        return "Stock Price ('{}', '{}', '{}', '{}')".format(self.date, self.vol, self.adj_close, self.query)

def update_daily_stock_price(sp500_file):
    with open(sp500_file, 'rb') as f:
        tickers = pickle.load(f)
    
    date = str(dt.datetime.now().date()).split('-')
    
    start = dt.datetime(int(date[0]), int(date[1]), int(date[2])-1)
    end = dt.datetime(int(date[0]), int(date[1]), int(date[2]))

    stock_list = []
    for ticker in tqdm(tickers):
        try:
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            stock_data = df.values[-1]

            date = stock_data[0].to_pydatetime()
            stock_list.append(StockPrice(date, stock_data[1], stock_data[2], stock_data[3], stock_data[4], stock_data[5], stock_data[6], ticker))
        
        except:
            print('Get Nothing.')
            
    db.session.add_all(stock_list)
    db.session.commit()



def save_stock_price_to_db(filename):
    df = pd.read_csv(filename)
    # date = dt.datetime.strptime(row['date'], '%Y-%m-%d')
    df['Date'] = df.Date.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
    _lst = []
    for index, row in df.iterrows():
        _lst.append(StockPrice(row['Date'], row['High'], row['Low'], row['Open'], row['Close'], row['Volume'], row['Adj Close'], row['query']))
    db.session.add_all(_lst)
    db.session.commit()