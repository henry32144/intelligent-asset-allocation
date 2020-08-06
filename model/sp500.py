from database.database import db
from database.tables.price import StockPrice

import numpy as np
import pandas as pd
import datetime as dt

class SP500(object):
    def __init__(self, tick='^GSPC'):
        self.tick = tick

    def read_stock_file(self):
        stock_data = StockPrice.find_all_by_query(comp=self.tick)
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

        return train_df, test_df


    def get_backtest_result(self):
        _, test_df = self.read_stock_file()
        sp500_price = test_df['adj_close']

        log_return = np.log(sp500_price) - np.log(sp500_price.shift(1)).values
        log_return = log_return[1:]

        sp500_values = []
        # all_return = []
        total_value = 1

        for i in range(len(log_return)):
            curr_log_return = log_return[i]
            total_value = total_value * (1 + curr_log_return)
            
            sp500_values.append(total_value)
            # all_return.append(curr_log_return)

        sp500_values.insert(0, 1)

        return sp500_values