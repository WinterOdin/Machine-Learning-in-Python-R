import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_pol.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)
a = regressor.predict([[6.5]])
print(a)
