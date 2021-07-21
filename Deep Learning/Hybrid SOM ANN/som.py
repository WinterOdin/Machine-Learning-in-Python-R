import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import  MiniSom
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=15, sigma = 1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration=100)

from pylab import bone,colorbar, pcolor, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors  = ['r', 'g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+ 0.5,
    w[1]+ 0.5,
    markers[y[i]],
    markeredgecolor = colors[y[i]],
    markerfacecolor = "None",
    markersize = 10,
    markeredgewidth = 2 )
show()

maps = som.win_map(X)
fake = np.concatenate((maps[(5,3)],maps[(8,3)]), axis=0)
fake = sc.inverse_transform(fake)



dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#encoding label female/male
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# encoding categorical data, unike ID's for sting data like germany is 1.0 0.0 0.0
#change nummber in ct to column that yopu wanna encode
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
X  = np.array(ct.fit_transform(X))

# spliting dataset into Test and Traning set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# feature scaling minimalizing domination 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

#initializing ann
ann = tf.keras.models.Sequential()
#adding first hidden layer 
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
#compiling ann 
ann.compile(optimizer="adas", loss="binary_crossentropy", metrics=["accuracy"])
#traning ann 
ann.fit(X_train, y_train, batch_size=32, epochs=100)