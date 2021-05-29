from datetime import timedelta, datetime

import numpy as np

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, GRU, SimpleRNN

from app.common import mongo


def insert_prediction_result(algo, ticker, date: datetime, closes, prediction):
    closes = closes.T.tolist()[0]
    prediction = prediction.T.tolist()[0]

    for i in range(len(closes)):
        mongo.client[f'{ticker}_{algo}_prediction'].insert(
            {"Date": date, "Close": closes[i], "Prediction": prediction[i]})
        date = date + timedelta(days=1)


def normalize(df: DataFrame):
    cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    scalar = MinMaxScaler()
    norm_data = scalar.fit_transform(df[cols])
    norm_df = pd.DataFrame(norm_data)
    norm_df.columns = cols
    scalar.fit_transform(df[['Close']])

    return norm_df, scalar


def generate_dataset(x, y, window_size=10):
    data_x = []
    data_y = []

    for i in range(len(y) - window_size):
        data_x.append(np.array(x.iloc[i:i + window_size]))
        data_y.append(np.array(y.iloc[i + window_size]))

    return np.array(data_x), np.array(data_y)


def make_train_test_dataset(data_x, data_y):
    train_size = int(len(data_y) * 0.7)
    train_x = np.array(data_x[0: train_size])
    train_y = np.array(data_y[0: train_size])

    test_x = np.array(data_x[train_size: len(data_x)])
    test_y = np.array(data_y[train_size: len(data_y)])

    return train_x, test_x, train_y, test_y


def get_first_prediction_date(df):
    train_size = int(len(df) * 0.7)
    period = np.array(df[['Date']])[train_size: len(df)]
    return period[0][0]


def build_model(algo, train_x):
    model = Sequential()
    if algo == 'rnn':
        model.add(SimpleRNN(units=10, activation='relu', return_sequences=True,
                            input_shape=(train_x.shape[1], train_x.shape[2])))
    elif algo == 'lstm':
        model.add(LSTM(units=10, activation='relu', return_sequences=True,
                       input_shape=(train_x.shape[1], train_x.shape[2])))
    else:
        model.add(GRU(units=10, activation='relu', return_sequences=True,
                      input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.1))

    if algo == 'rnn':
        model.add(SimpleRNN(units=10, activation='relu'))
    elif algo == 'lstm':
        model.add(LSTM(units=10, activation='relu'))
    else:
        model.add(GRU(units=10, activation='relu'))

    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.summary()

    return model


def run():
    tickers = mongo.client['tickers'].find()
    for item in list(tickers):
        data = mongo.client[f'{item.get("ticker")}_stocks'].find({}, {'_id': False})
        df = pd.DataFrame.from_records(data)

        norm_df, scalar = normalize(df)
        x = norm_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        y = norm_df[['Close']]
        data_x, data_y = generate_dataset(x, y)

        train_x, test_x, train_y, test_y = make_train_test_dataset(data_x, data_y)
        # train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3)
        closes = scalar.inverse_transform(test_y)
        for algo in ('rnn', 'lstm', 'gru'):
            model = build_model(algo, train_x)

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(train_x, train_y, epochs=60, batch_size=30)
            prediction = model.predict(test_x)
            prediction = scalar.inverse_transform(prediction)

            first_date = get_first_prediction_date(df)
            insert_prediction_result(algo, item.get("ticker"),
                                     pd.to_datetime(first_date),
                                     closes=closes,
                                     prediction=prediction)
