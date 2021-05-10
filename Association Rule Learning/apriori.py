import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori as ap
from csv import reader

with open('Market_Basket_Optimisation.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    transactions = list(csv_reader)
    

rules = ap(transactions = transactions, min_support=0.003, min_confidence = 0.2, min_lift = 3, min_length=2, max_length=2)
rules = list(rules)

def create(results):
    left  = [tuple(result[2][0][0])[0] for result in results]
    right = [tuple(result[2][0][1])[0] for result in results]
    supp  = [result[1] for result in results]
    conf  = [result[2][0][2] for result in results]
    lift  = [result[2][0][3] for result in results]
    return list(zip(left,right,supp,conf,lift))
resultDF = pd.DataFrame(create(rules), columns=['left','right','support','confidence','lift'])
resultDF = resultDF.nlargest(n = 10, columns = 'lift')
print(resultDF)