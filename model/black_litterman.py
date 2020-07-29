from database.database import db
from database.tables.price import StockPrice

from tqdm import tqdm
import datetime as dt
import numpy as np
import pandas as pd
import scipy
from pandas_datareader import data
import matplotlib.pyplot as plt


class Black_Litterman(object):
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

        return train_df, test_df


    def prepare_input(self):
        # 1. Collect user selected tickers' stock price
        all_data_df = pd.DataFrame({})
        time_step = 180

        for ticker in self.selected_tickers:
            train_df, test_df = self.read_stock_file(ticker)
            # axis=0 -> combine vertically
            dataset_total = pd.concat([train_df, test_df], axis=0) 
            inputs = dataset_total[len(dataset_total)-len(test_df)-time_step:]['adj_close']
            all_data_df[ticker] = inputs
        
        # 2. Prepare main markowitz inputs
        # 2-1. price_df -> price_list
        prices_list = []
        for i in range(time_step, len(all_data_df)):
            price_t = all_data_df[i-time_step:i].T
            prices_list.append(price_t)

        # 2-2. get market capitalization
        end = dt.datetime(2016, 12, 31) 
        market_caps = list(data.get_quote_yahoo(self.selected_tickers, end)['marketCap'])

        return prices_list, market_caps


    def assets_historical_returns_covariances(self, prices_list):
        all_exp_returns = []
        all_covars = []
        
        for prices in prices_list:
            prices = np.array(prices)
            rows, cols = prices.shape
            returns = np.empty([rows, cols - 1])

            for r in range(rows):
                for c in range(cols-1):
                    p0, p1 = prices[r, c], prices[r, c+1]
                    returns[r, c] = (p1/p0) - 1

            # calculate returns
            exp_returns = np.array([])
            for r in range(rows):
                exp_returns = np.append(exp_returns, np.mean(returns[r]))

            # calculate covariances
            covars = np.cov(returns)

            # annualize returns, covariances
            exp_returns = (1 + exp_returns) ** (365.25) - 1
            covars = covars * (365.25)
            
            all_exp_returns.append(exp_returns)
            all_covars.append(covars)
        
        return all_exp_returns, all_covars


    def portfolioMean(self, W, R):
        return sum(R * W)


    def portfolioVar(self, W, C):
        return np.dot(np.dot(W, C), W)


    def portfolioMeanVar(self, W, R, C):
        return self.portfolioMean(W, R), self.portfolioVar(W, C)


    def calculate_reverse_pi(self, W, all_R, all_C):
        reverse_pi = []
        rf = 0.015

        for i in range(len(all_R)):
            mean, var = self.portfolioMeanVar(W, all_R[i], all_C[i])
            lmb = (mean - rf) / var  # Calculate risk aversion
            temp = np.dot(lmb, all_C[i])
            Pi = np.dot(temp, W)
            reverse_pi.append(Pi)

        return reverse_pi

    def get_predicted_Q(self):
        


    def get_all_weights(self):
        prices_list, W = self.prepare_input() # W: market capitalization
        all_R, all_C = self.assets_historical_returns_covariances(prices_list)
        reverse_pi = self.calculate_reverse_pi(W, all_R, all_C)



