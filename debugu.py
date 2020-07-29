import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import datetime as dt
yf.pdr_override()
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 7, 1)
# df = web.DataReader('NFLX','yahoo', start, end)
df = web.get_data_yahoo('NFLX' ,start,end)
print(df.head())