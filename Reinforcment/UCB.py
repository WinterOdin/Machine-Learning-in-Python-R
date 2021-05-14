import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 1000000
D = 10
ads_selected = []
numbers_selection = [0] * D
sum_rewards = [0] * D 
total_reward = 0