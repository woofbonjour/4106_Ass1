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

def knapsack_value(solution, prices):
    return sum([prices[i] for i in solution])

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

    curr_solution = []
    curr_solution_price = 0

    # Extracting data from row
    weights = data.iloc[0]
    prices = data.iloc[1]
    capacity = data.iloc[2]

    n = len(weights) # Number of blocks available
    currTemp = initial_temperature # Set current temperature

    # Initialize with a random solution
    for i in range(n):
        if random.random() < 0.5:
            curr_solution.append(i)

    # For max iterations
    for iteration in range(N):
        # Generate a neighbor solution by cloning the current solution
        neighbor_solution = list(curr_solution)
        
        # Randomly decide whether to add or remove an item
        if random.random() < 0.5:
            # Try to add an item with the highest value-to-weight ratio
            best_candidate = None
            best_ratio = 0
            for i in range(n):
                if i not in neighbor_solution:
                    ratio = prices[i] / weights[i]
                    if weights[i] + knapsack_weight(neighbor_solution, weights) <= capacity and ratio > best_ratio:
                        best_candidate = i
                        best_ratio = ratio
            if best_candidate is not None:
                neighbor_solution.append(best_candidate)
        else:
            # Try to remove an item with the lowest value-to-weight ratio
            worst_candidate = None
            worst_ratio = float('inf')
            for i in neighbor_solution:
                ratio = prices[i] / weights[i]
                if ratio < worst_ratio:
                    worst_candidate = i
                    worst_ratio = ratio
            if worst_candidate is not None:
                neighbor_solution.remove(worst_candidate)

        # Calculate the value and weight of the neighbor solution
        neighbor_value = knapsack_value(neighbor_solution, prices)
        neighbor_weight = knapsack_weight(neighbor_solution, weights)

        # Accept or reject the neighbor solution based on acceptance probability
        if neighbor_weight <= capacity and acceptance_probability(curr_solution_price, neighbor_value, currTemp) > random.random():
            curr_solution = neighbor_solution
            curr_solution_price = neighbor_value

        if curr_solution_price > best_solution_price:
            best_solution = curr_solution
            best_solution_price = curr_solution_price

        # Reduce the temperature
        currTemp *= cooling_rate

    return best_solution_price, best_solution
  


##################################################################3
#'''
solutions_sa = []
printnow = True
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = simulated_annealing(row, N = 50, initial_temperature=1, cooling_rate=0.95)
    solutions_sa.append(1 if target == solution else 0)

# Accuracy
print("Simulated Annealing Accuracy is", np.mean(solutions_sa))
#'''


'''
elif len(neighbor_solution)<n:
    # Choose random block to add to knapsack
    while True:
        item_to_add = random.randint(0, n - 1)
        if item_to_add not in neighbor_solution:
            neighbor_solution.append(item_to_add)
            break
                    
'''