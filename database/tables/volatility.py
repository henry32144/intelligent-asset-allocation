from database.database import db
from database.tables.price import StockPrice

import numpy as np


class Volatility(object):
    def __init__(self, tick):
        self.tick = tick

    def read_stock_file(self):
        stock_data = StockPrice.find_all_by_query(comp=self.tick)
        stock_close_price = [row.adj_close for row in stock_data]

        return stock_close_price

    def calculate_sharpe_ratio(self):
        stock_close_price = self.read_stock_file()
        risk_free = 0.015

        arr_values = np.array(stock_close_price)
        numerator = np.mean(arr_values - risk_free)
        denominator = np.std(arr_values - risk_free)
        sharpe_ratio = numerator / denominator

        return sharpe_ratio