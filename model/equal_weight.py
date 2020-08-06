from database.database import db
from database.tables.price import StockPrice

from tqdm import tqdm
import datetime as dt
import numpy as np
import pandas as pd

class EqualWeight(object):
    def __init__(self, selected_tickers):
        self.selected_tickers = selected_tickers

    def read_stock_file(self, tick):
        # Use the Flask-SQLAlchemy to query our data from database
        stock_data = StockPrice.find_all_by_query(comp=tick)
        
        date_ = []
        high = []
        low = []
        open_ = []
        adj_close = []
        vol = []
        
        # Store/Split the data into train & test dataframe
        for row in stock_data:
            date = dt.datetime.strptime(str(row.date), '%Y-%m-%d')
            date_.append(date)
            high.append(row.high)
            low.append(row.low)
            open_.append(row.open_)
            adj_close.append(row.adj_close)
            vol.append(row.vol)

        df = pd.DataFrame({
            'date': date_,
            'high': high,
            'low': low,
            'open': open_,
            'adj_close': adj_close,
            'vol': vol
        })
        df.set_index('date', inplace=True)

        # split dataframe into train & test part
        train_df, test_df = df['2012-01-01': '2016-12-31'], df['2017-01-01': '2020-06-30']

        self.test_df = test_df

        return train_df, test_df


    def get_backtest_result(self):
        all_test_df = pd.DataFrame({})

        for tick in self.selected_tickers:
            _, test_df = self.read_stock_file(tick)
            price = test_df['adj_close']
            log_return = np.log(price) - np.log(price.shift(1))
            all_test_df = pd.concat([all_test_df, log_return], axis=1)

        all_test_df = all_test_df.iloc[1:, :]
        all_test_df.columns = self.selected_tickers

        n_tickers = len(self.selected_tickers)
        equal_weights = np.array([1/n_tickers]*n_tickers)
        length = len(all_test_df)
        all_equal_weights = np.concatenate([[equal_weights]] * length, axis=0)
    
        total_value = 1
        all_values = []
        
        for i in range(length):
            portfolio_weights = all_equal_weights[i]
            returns = all_test_df.iloc[i, :].values
            portfolio_return = sum(portfolio_weights * returns)
            total_value = total_value * (1+portfolio_return)
        
            all_values.append(total_value)

        date = self.test_df.reset_index()['date']
        date = date.dt.strftime('%Y-%m-%d')
        date = date.values[1:]
         
        return date, all_values
        
