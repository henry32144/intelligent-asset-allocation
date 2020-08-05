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

    def calculate_std(self):
        stock_close_price = self.read_stock_file()
        arr_values = np.array(stock_close_price)
        std = np.std(arr_values)

        return std