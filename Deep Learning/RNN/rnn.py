import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("stock_price_google.csv")
training_dataset = dataset.iloc[:,1:2].values

#feature scaling 
from sklearn.preprocessing import  MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set = sc.fit_transform(training_dataset)

X_train = []
y_train = []

for x in range(60,1258):
    X_train.append(training_set[x-60:i,0])
    y_train.append(training_set[x:x,0])

X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import  Dropout

#init RNN
regressor = Sequential()

#adding first layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#adding 2 layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#adding 3 layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#adding 4 layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#ut layer
regressor.add(Dense(units=1))