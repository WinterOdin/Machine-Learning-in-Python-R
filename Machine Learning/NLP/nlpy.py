import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#stemming conwerts words like I'loved this to I love this 

dataset = pd.read_csv('review.tsv', delimiter='\t', quoting=3)
corpus  = []
for x in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][x])
    review = review.lower()
    review = review.split()
    #stemming
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    corpus.append(review)


#bag wof words
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X   = vec.fit_transform(corpus).toarray()
y   = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


#traning 
from sklearn.svm import SVC
classyfier = SVC(kernel ='rbf')
classyfier.fit(X_train, y_train)



y_pred = classyfier.predict(X_test)
y_pred_len = len(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
#accuracy
accur = accuracy_score(y_test,y_pred)
print(accur)