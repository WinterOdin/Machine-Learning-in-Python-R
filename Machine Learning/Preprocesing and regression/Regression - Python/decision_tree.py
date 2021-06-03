import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_pol.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

regressor.predict([[6.5]])

X_grind = np.arange(min(X), max(X), 0.1)
X_grind = X_grind.shape((len(X_grind),1 ))
plt.scatter(X,y, color="red")
plt.plot(X_grind, regressor.predict(X_grind), color="blue")
plt.show()