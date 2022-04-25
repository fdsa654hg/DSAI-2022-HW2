import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, ReLU
from keras.callbacks import EarlyStopping
import random
import tensorflow as tf
import os
import math

from keras_radam import RAdam

from sklearn.preprocessing import StandardScaler

# 固定seed
seed_value = 87
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

# 參考前面幾天的資料
look_before_days = 7

def create_data(data, past = 7, future = 7):
    x_train = []
    y_train = []
    for i in range(0, len(data)-past-future+1):
        x = data.iloc[i:i+past][['open', 'high', 'low', 'close']]
        y = data['difference'].iloc[i+past:i+past+future]

        x_train.append(x)
        y_train.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

def build_model(shape):
    model = Sequential()
    model.add(LSTM(20, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(LSTM(20))
    model.add(Dense(units = 20))
    model.add(Dropout(0.2))
    model.add(Dense(units = 10))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(RAdam(), loss="binary_crossentropy", metrics=['accuracy'])
    model.summary()
    return model

def calculate_earnings(outputs, data):
    earn = 0.0
    stock = 0
    buy = 0
    for i, j in zip(outputs, data):
        if i == 1:
            stock += 1
            buy = j
            earn -= buy
        elif i == -1:
            stock -= 1
            buy = j
            earn += buy
    if stock >= 1:
        for i in range(stock):
            earn += data[-1]
    elif stock < 0:
        for i in range(abs(stock)):
            earn -= data[-1]
    stock = 0
    return earn


if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    train = pd.read_csv(args.training, names=['open', 'high', 'low', 'close'])
    test = pd.read_csv(args.testing, names=['open', 'high', 'low', 'close'])
    all_data = train.append(test)

    # difference 是明天開盤價的漲幅(明天開盤價-今天收盤價)
    d_list = []
    for i in range(len(train)-1):
        d_list.append(float(train.iloc[i+1:i+2][['open']].values[0]) - float(train.iloc[i:i+1][['close']].values[0]))
    train['difference'] = pd.DataFrame(d_list)
    train.loc[len(train['difference'])-1, 'difference'] = float(test.iloc[0:1][['open']].values[0]) - float(train.iloc[len(train['difference'])-1:len(train['difference'])][['close']].values[0])

    d_list = []
    for i in range(len(test)-1):
        d_list.append(float(test.iloc[i+1:i+2][['open']].values[0]) - float(test.iloc[i:i+1][['close']].values[0]))
    test['difference'] = pd.DataFrame(d_list)

    # difference>0設成1，<0設成0
    ss = StandardScaler()
    train_scaled = ss.fit_transform(train)
    train_scaled = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    train_scaled['difference'] = train['difference']>=0
    train_scaled['difference'] = train_scaled['difference'].astype(int)
    train_scaled.tail()

    test_scaled = ss.fit_transform(test)
    test_scaled = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)
    test_scaled['difference'] = test['difference']>=0
    test_scaled['difference'] = test_scaled['difference'].astype(int)
    test_scaled.tail()

    # training
    x_train, y_train = create_data(train_scaled,look_before_days, 1)
    callback = EarlyStopping(monitor="val_accuracy", patience=400, verbose=1, mode="auto")
    history = build_model(x_train.shape)
    # batch_size大小設為training set的三分之一
    history.fit(x_train, y_train, batch_size=len(x_train)//3, validation_split=0.2, epochs = 1000, callbacks=[callback])

    test_length = len(test)
    all_data = train_scaled.append(test_scaled)
    x_test, y_test = create_data(all_data, look_before_days, 1)
    x_test = x_test[-test_length+1:]
    y_test = y_test[-test_length+1:]
    actions = []

    output_file = open(args.output, 'w')

    action = 0
    stock = 0  # 持股數量

    # testing
    for i in range(test_length-1):
        
        tmp = x_test[i].reshape(1, look_before_days, -1)
        tmp = history.predict(tmp)
        tmp = tmp[0][0]
        if tmp>=0.55 and stock<1:
            action = 1
            stock += 1
        elif tmp<=0.45 and stock>-1:
            action = -1
            stock -= 1
        else:
            action = 0

        actions.append(action)
        output_file.write(str(action))
        if i < test_length-2:
            output_file.write('\n')

    output_file.close()
    outputs = pd.read_csv('output.csv', header=None)
    print('earn:' + str(calculate_earnings(outputs[0].tolist(), test['open'].tolist())))