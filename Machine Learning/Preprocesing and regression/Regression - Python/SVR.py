import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('svr-data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
x_1 = X
y_1 = y
# making 2D y

y = y.reshape(len(y),1)


"""
Musimy robić feature scaling/standaryzacje jeśli x i y różnią się o siebie bardzo duzo
"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X  = sc_X.fit_transform(X)
y  = sc_Y.fit_transform(y)

from sklearn.svm import SVR
reqressor = SVR(kernel='rbf')
reqressor.fit(X,y)


a = sc_Y.inverse_transform((reqressor.predict(sc_X.transform([[6.5]]))))


#plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color='red')
#plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(reqressor.predict(X)), color="blue")
#plt.show()


plt.scatter(x_1, y_1, color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(reqressor.predict(X)), color="blue")
plt.show()