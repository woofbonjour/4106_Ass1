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

##-------------------


def knapsack_value(solution, values):
    return sum([values[i] for i in solution])

def knapsack_weight(solution, weights):
    return sum([weights[i] for i in solution])

def acceptance_probability(current_value, new_value, currTemp):
    if new_value > current_value:
        return 1.0
    return math.exp((new_value - current_value) / currTemp)

##-------------------


def simulated_annealing(data, N, initial_temperature, cooling_rate):

    # Base case for when there are no solutions
    best_solution = []
    best_solution_price = 0

    # Extracting data from row
    weight = data.iloc[0]
    prices = data.iloc[1]
    capacity = data.iloc[2]

    n = len(weight) # Number of blocks available
    currTemp = initial_temperature # Set current temperature

    # Initialize with a random solution
    for i in range(n):
        if random.random() < 0.5:
            best_solution.append(i)

    # For max iterations
    for iteration in range(N):
        # Generate a neighbor solution by cloning the current solution
        neighbor_solution = list(best_solution)
        # Then choose randomly whether to add or remove a block
        if random.random() < 0.5 and len(neighbor_solution) > 0:
            # Choose random block to remove from the knapsack
            item_to_remove = random.randint(0, len(neighbor_solution) - 1)
            neighbor_solution.pop(item_to_remove)
        else:
            # Choose random block to add to knapsack
            item_to_add = random.randint(0, n - 1)
            if item_to_add not in neighbor_solution:
                neighbor_solution.append(item_to_add)

        # Calculate the value and weight of the neighbor solution
        neighbor_value = knapsack_value(neighbor_solution, prices)
        neighbor_weight = knapsack_weight(neighbor_solution, weight)

        # Accept or reject the neighbor solution based on acceptance probability
        if neighbor_weight <= capacity and acceptance_probability(best_solution_price, neighbor_value, currTemp) > random.random():
            best_solution = neighbor_solution
            best_solution_price = neighbor_value

        # Reduce the temperature
        currTemp *= cooling_rate

    return best_solution_price, best_solution
  


##################################################################3
#'''
solutions_sa = []
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = simulated_annealing(row, N = 10, initial_temperature=1, cooling_rate=0.95)
    solutions_sa.append(1 if target == solution else 0)

# Accuracy
print("Simulated Annealing Accuracy is", np.mean(solutions_sa))
#'''