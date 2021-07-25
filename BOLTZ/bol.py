import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies   = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding="latin-1")
users    = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding="latin-1")
ratings  = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding="latin-1")

traning = pd.read_csv('ml-100k/u1.base', delimiter='\t')
traning = np.array(traning, dtype='int')

test = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test = np.array(test, dtype='int')

nb_users  = int(max(max(traning[:,1]), max(test[:,1])))
nb_movies = int(max(max(traning[:,0]), max(test[:,0])))

def convert(data):
    new_data  = []
    for usr_id in range(0, nb_users+1):
        id_movies = data[:,1][data[:,0] == usr_id]
        id_ratings = data[:,2][data[:,0] == usr_id]
        ratings = np.zeros(nb_movies)
        ratings[id_ratings - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

traning = convert(traning)
test    = convert(test)

traning = torch.FloatTensor(traning)
test    = torch.FloatTensor(test)

traning[traning == 0 ] = -1
traning[traning == 1 ] = 0
traning[traning == 2 ] = 0
traning[traning == 3]  = 1
test[test == 0 ] = -1
test[test == 1 ] = 0
test[test == 2 ] = 0
test[test == 3 ] = 1