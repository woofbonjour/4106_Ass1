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

def acceptance_probability(current_value, new_value, temperature):
    if new_value > current_value:
        return 1.0
    return math.exp((new_value - current_value) / temperature)

##-------------------


def simulated_annealing(data, N, initial_temperature, cooling_rate):

    # Base case for when there are no solutions
    best_solution = [0]*5
    best_solution_price = 0

    # Extracting data from row
    weight = data.iloc[0]
    prices = data.iloc[1]
    capacity = data.iloc[2]

    n = len(weight) # Number of blocks available
    currTemp = initial_temperature # Set current temperature

    # Sorting data by weight
    # Combine the two lists into a list of tuples (element, index)
    combined_list = [(weight[i], prices[i], i) for i in range(len(weight))]

    # Sort the combined list based on weight in ascending order
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0])

    # Extract the sorted values from the combined list
    sorted_weight = [x[0] for x in sorted_combined_list]
    sorted_prices = [x[1] for x in sorted_combined_list]

    
    
    


    return best_solution_price, best_solution

    # Initialize with a random solution
    for i in range(n):
        if random.random() < 0.5:
            current_solution.append(i)

    for iteration in range(max_iterations):
        # Generate a neighbor solution by adding or removing an item
        neighbor_solution = list(current_solution)
        if random.random() < 0.5 and len(neighbor_solution) > 0:
            item_to_remove = random.randint(0, len(neighbor_solution) - 1)
            neighbor_solution.pop(item_to_remove)
        else:
            item_to_add = random.randint(0, n - 1)
            if item_to_add not in neighbor_solution:
                neighbor_solution.append(item_to_add)

        # Calculate the value and weight of the neighbor solution
        neighbor_value = knapsack_value(neighbor_solution, values)
        neighbor_weight = knapsack_weight(neighbor_solution, weights)

        # Accept or reject the neighbor solution based on acceptance probability
        if neighbor_weight <= capacity and acceptance_probability(current_value, neighbor_value, temperature) > random.random():
            current_solution = neighbor_solution
            current_value = neighbor_value

        # Reduce the temperature
        temperature *= cooling_rate

    return current_solution, current_value
  


##################################################################3
#'''
solutions_sa = []
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = simulated_annealing(row, N = 10, initial_temperature=1, cooling_rate=0.95)
    solutions_sa.append(1 if target == solution else 0)

# Accuracy
print("Simulated Annealing Accuracy is", np.mean(solutions_sa))'
#'''