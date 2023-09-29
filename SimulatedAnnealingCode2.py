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

def acceptance_probability(current_value, new_value, currTemp):
    if new_value > current_value:
        return 1.0
    return math.exp((new_value - current_value) / currTemp)

def accept(current_energy, new_energy, currTemp):
    if new_energy>current_energy:
        return True
    if random.random() < math.exp(abs(current_energy-new_energy)/currTemp):
        return True
    return False

##-------------------


def simulated_annealing(data, N, initial_temperature, cooling_rate):
    # Base case for when there are no solutions
    curr_solution = [0]*5
    curr_solution_price = 0

    # Extracting data from row
    weights = data.iloc[0]
    prices = data.iloc[1]
    capacity = data.iloc[2]

    n = 5 # len(weights) # Number of blocks available
    currTemp = initial_temperature # Set current temperature

    # Initialize with a random solution
    for i in range(n):
        if random.random() < 0.5:
            curr_solution[i] = 1^curr_solution[i]

    # For max iterations
    for iteration in range(N):
        # Generate a neighbor solution by cloning the current solution
        neighbor_solution = list(curr_solution)
        # Choose one block at random to remove or add
        item_to_alter = random.randint(0,n-1)
        neighbor_solution[item_to_alter] = 1^neighbor_solution[item_to_alter]

        # Calculate the value and weight of the neighbor solution
        neighbor_price = sum(x*y for x, y in zip(neighbor_solution, prices))
        neighbor_weight = sum(x*y for x, y in zip(neighbor_solution, weights))

        # Accept or reject the neighbor solution based on acceptance probability
        if neighbor_weight <= capacity and accept(curr_solution_price,neighbor_price,currTemp):
        # acceptance_probability(best_solution_price, neighbor_price, currTemp) > random.random():
            curr_solution = neighbor_solution
            curr_solution_price = neighbor_price

        # Reduce the temperature
        currTemp *= cooling_rate

    best_solution = [x*y for x, y in zip(curr_solution, weights)]
    best_solution_price = sum([x*y for x, y in zip(curr_solution, prices)])

    return best_solution_price, best_solution
  


##################################################################3
#'''
solutions_sa = []
printnow = True
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = simulated_annealing(row, N = 10, initial_temperature=1, cooling_rate=0.95)
    solutions_sa.append(1 if target == solution else 0)
    
    if printnow and (indexes == row['Best picks']):
        print(row)
        printnow = False

# Accuracy
print("Simulated Annealing Accuracy is", np.mean(solutions_sa))
#'''