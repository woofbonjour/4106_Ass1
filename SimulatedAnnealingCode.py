import pandas as pd
import itertools
import numpy as np

url = 'https://raw.githubusercontent.com/woofbonjour/4106_Ass1/main/KnapsackDataset.csv'

dataset = pd.read_csv(url)
dataset.columns
dataset.head(10)

def string_to_list(string):

  string_list = string.strip('[]').split()

  float_list = [float(element) for element in string_list]

  return float_list

#Ignore the warning messages.
dataset = dataset.dropna()

dataset.Weights = dataset.Weights.apply(lambda x : string_to_list(x))
dataset.Prices = dataset.Prices.apply(lambda x : string_to_list(x))
dataset['Best picks'] = dataset['Best picks'].apply(lambda x : string_to_list(x))



###############################################################

import random
import math


def simulated_annealing(data, N, initial_temperature, cooling_rate):



  return best_solution_price, best_solution



##################################################################3

solutions_sa = []
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = simulated_annealing(row, N = 10, initial_temperature=1, cooling_rate=0.95)
    solutions_sa.append(1 if target == solution else 0)

# Accuracy
print("Simulated Annealing Accuracy is", np.mean(solutions_sa))