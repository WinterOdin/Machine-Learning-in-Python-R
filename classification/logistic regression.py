import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('customer.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#traning 
from sklearn.linear_model import LogisticRegression
classyfier = LogisticRegression()
classyfier.fit(X_train, y_train)


"""
Showing data in two columns

left - predicted
right - real
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""
y_pred = classyfier.predict(X_test)
y_pred_len = len(y_pred)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
#accuracy
accur = accuracy_score(y_test,y_pred)
print(accur)