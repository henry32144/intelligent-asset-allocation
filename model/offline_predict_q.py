import sys
sys.path.insert(1, '..')
import pickle
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import datetime as dt
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers

from database.database import db

from database.tables.price import StockPrice
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine('sqlite:///../database/database.db', echo = True)
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
Session = sessionmaker(bind = engine)

def company_name2dict():
    with open('../database/tables/ticker_name.txt', 'r') as f:
        _dict = {}
        lines = f.readlines()
        for l in lines:
            c = l.split('\t')
            _dict[c[1].replace('\n','')] = c[0]
    f.close()
    return _dict

def to_model_input(time_step, dataset, target_col_idx):
    X = []
    y = []

    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, :])
        y.append(dataset[i, target_col_idx])
        
    return np.array(X), np.array(y)

def train_model(X_train, y_train, epochs, batch_size):
    model = Sequential()
    model.add(GRU(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 5)))
    model.add(GRU(units = 100, return_sequences = True))
    model.add(GRU(units = 100, return_sequences = True))
    model.add(GRU(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    # Compiling
    optimizer = optimizers.Adam()
    model.compile(optimizer = optimizer , loss = 'mean_squared_error')

    train_history = model.fit(X_train, y_train, epochs=epochs, \
                                    batch_size=batch_size, verbose=1)

    return model


def get_prediction(model, X_test, test_df, min_max_scaler):
    predicted_price_scaler = model.predict(X_test)
    real_stock_price = test_df.values[:, 3]
    real_stock_price_scaler = min_max_scaler.transform(test_df.values)

    # Convert the predicted price_scaler back
    real_stock_price_scaler[:, 3] = predicted_price_scaler.flatten()
    predicted_price = min_max_scaler.inverse_transform(real_stock_price_scaler)[:, 3]
    
    return predicted_price, real_stock_price

def get_predicted_Q():
    # with open('./sp500tickers.pkl', 'rb') as f:
    #     tickers = pickle.load(f)
    sp500_wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(sp500_wiki_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_='wikitable sortable')
    trs = table.find_all('tr')

    symbols = []
    company_names = []
    industries = []

    for idx, tr in enumerate(trs):
        if idx != 0:
            tds = tr.find_all('td')
            symbols.append(tds[0].text.replace('\n',''))
            company_names.append(tds[1].text.replace('\n',''))
            industries.append(tds[3].text.replace('\n',''))

    stock_data_df = pd.DataFrame({
        'symbol': symbols,
        'company': company_names,
        'industry': industries
    })

    tickers = []
    company = []
    industry = []
    stock_dict = company_name2dict()

    sym = list(stock_dict.keys())
    com = list(stock_dict.values())

    for idx, row in stock_data_df.iterrows():
        if row[0] in sym:
            tickers.append(row[0])
            company.append(stock_dict[row[0]])
            industry.append(row[2])

    Q_predict_dict = defaultdict(list)
    real_price_dict = defaultdict(list)

    for tick in tqdm(tickers):
        predicted_price, real_stock_price, test_df = predict_Q(tick)
        Q_predict_dict[tick].append(predicted_price)
        real_price_dict[tick].append(real_stock_price)

    with open('predict_dict', 'wb') as f:
        pickle.dump(Q_predict_dict, f)

    with open('real_price_dict', 'wb') as f:
        pickle.dump(real_price_dict, f)

    return ''

def get_stock_price_offline(tick):
    session = Session()

    stock_default_list = defaultdict(list)

    stock_default_list[tick] = []
    
    # Use the Flask-SQLAlchemy to query our data from database
    stock_data = session.query(StockPrice).filter(StockPrice.comp==tick).all()

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
    
    # We need to standardize the input before putting them into the model
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled  = min_max_scaler.fit_transform(train_df.values)
    time_step = 180
    target_col_idx = 3

    # Get the trainset part
    X_train, y_train = to_model_input(time_step, train_scaled, target_col_idx)

    # Get the testset part
    dataset_total = pd.concat([train_df, test_df], axis=0)
    testing_inputs = dataset_total[len(dataset_total)-len(test_df)-time_step:]
    testing_scaled = min_max_scaler.transform(testing_inputs)
    X_test, y_test = to_model_input(time_step, testing_scaled, target_col_idx)

    stock_default_list[tick].append(X_train)
    stock_default_list[tick].append(y_train)
    stock_default_list[tick].append(X_test)
    stock_default_list[tick].append(y_test)
    stock_default_list[tick].append(test_df)
    session.close()
    return test_df, stock_default_list, min_max_scaler

def predict_Q(tick):
    test_df, stock_default_list, min_max_scaler = get_stock_price_offline(tick)

    X_train = stock_default_list[tick][0]
    y_train = stock_default_list[tick][1]

    batch_size = 16
    epochs = 64

    model = train_model(X_train, y_train, epochs, batch_size)

    X_test = stock_default_list[tick][2]
    test_df = stock_default_list[tick][4]

    predicted_price, real_stock_price = get_prediction(model, X_test, test_df, min_max_scaler)
    
    return predicted_price, real_stock_price, test_df


if __name__ == "__main__":
    
    get_predicted_Q()
  
  