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

def generate_binary_combinations(length):
    combinations = []
    for i in range(2**length - 1, -1, -1):  # Start from 2^length - 1 and decrement
        binary_str = bin(i)[2:]  # Convert the integer to binary and remove the '0b' prefix
        binary_str = binary_str.zfill(length)  # Zero-fill to match desired length
        binary_list = [int(bit) for bit in binary_str]  # Convert the binary string to a list of integers
        combinations.append(binary_list)
    return combinations

# Generate all combinations of a binary list of length 5 in reverse order
combinations = generate_binary_combinations(5)

def gen_and_test(data):
    best_solution = [0]*5
    best_solution_price = 0

    weight = data.iloc[0]
    prices = data.iloc[1]
    capacity = data.iloc[2]
    
    for i in combinations: 
        new_weight_combination = [x*y for x, y in zip(i, weight)]
        new_price_combination = [x*y for x, y in zip(i, prices)]
        candidate_weight = sum(new_weight_combination)
        candidate_price = sum(new_price_combination)

        if candidate_weight <= capacity:
            best_solution = candidate_weight
            best_solution_price = candidate_price
            break

    return best_solution_price, best_solution


    # Generate all possible combinations of items
    for combination in all_combinations(items):
        total_weight = sum(item.weight for item in combination)
        total_value = sum(item.value for item in combination)

        # Check if the combination satisfies the weight constraint
        if total_weight <= capacity and total_value > best_value:
            best_solution = combination
            best_value = total_value

##################################################################3
solutions = []
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = gen_and_test(row)
    solutions.append(1 if target == solution else 0)


# Accuracy
print('Accuracy of best prices found is', np.mean(solutions))