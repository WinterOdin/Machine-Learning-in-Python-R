import numpy as np
import pandas as pd
import tensorflow as tf


dataset = pd.read_csv("modeling.csv")
X = dataset.iloc[:,3:-1].values
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

prediction_new = ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5 