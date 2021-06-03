#non linear relationships

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_pol.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
polyF       = PolynomialFeatures(degree=2)
X_poly      = polyF.fit_transform(X)
lin_reg_2   = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X,regressor.predict(X), color="blue")
plt.title("Linear reg 1  ")
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X,lin_reg_2.predict(polyF.fit_transform(X)), color="blue")
plt.title("Polynomial reg")
plt.show()

# predict new results with Linear Regression

regressor.predict([[6.5]])

## Predicting a new result with Polynomial Regression
lin_reg_2.predict([[6.5]])
