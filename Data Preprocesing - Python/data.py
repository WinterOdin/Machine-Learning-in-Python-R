import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Data.csv")
x    = data.iloc[:, :-1].values
y    = data.iloc[:,  -1].values

# replacing missing data via avg
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# encoding categorical data, unike ID's for sting data like germany is 1.0 0.0 0.0

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x  = np.array(ct.fit_transform(x))


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(y)

# spliting dataset into Test and Traning set 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# feature scaling minimalizing domination 


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform