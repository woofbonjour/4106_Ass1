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

    # Generate all possible combinations of block arrangments being in the knapsack (non-repetitive) in order of heaviest to lightest
    combinations = []
     # Start from all blocks being included (2^length - 1) and decrement
    for i in range(2**length - 1, -1, -1):
        binary_str = bin(i)[2:]  # Convert the integer to binary and remove the '0b' prefix
        binary_str = binary_str.zfill(length)  # Zero-fill to match desired length
        binary_list = [int(bit) for bit in binary_str]  # Convert the binary string to a list of integers
        combinations.append(binary_list)
    return combinations

# Generate all combinations of a binary list of length 5
combinations = generate_binary_combinations(5)

# Generate and test algorithm
def gen_and_test(data):
    # Base case for when there are no solutions
    best_solution = [0]*5
    best_solution_price = 0

    # Extracting data from row
    weight = data.iloc[0]
    prices = data.iloc[1]
    capacity = data.iloc[2]


    # Sorting data by weight
    # Combine the two lists into a list of tuples (element, index)
    combined_list = [(weight[i], prices[i], i) for i in range(len(weight))]

    # Sort the combined list based on weight in ascending order
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0])

    # Extract the sorted values from the combined list
    sorted_weight = [x[0] for x in sorted_combined_list]
    sorted_prices = [x[1] for x in sorted_combined_list]

    # For each possible combination of blocks in knapsack
    for i in combinations:
        # Generate the next heaviest candidate
        new_weight_combination = [x*y for x, y in zip(i, sorted_weight)]
        new_price_combination = [x*y for x, y in zip(i, sorted_prices)]
        candidate_weight = sum(new_weight_combination)
        candidate_price = sum(new_price_combination)

        # Test if the candadite is a valid solution
        if candidate_weight <= capacity:
            # If it is, break here and return this as the best solution
            best_solution = candidate_weight
            best_solution_price = candidate_price
            break
        # If not, then repeat with a new generated candidate

    return best_solution_price, best_solution
##################################################################3
solutions = []
for _, row in dataset.iterrows():
    target = row['Best price']
    solution, indexes = gen_and_test(row)
    solutions.append(1 if target == solution else 0)


# Accuracy
print('Accuracy of best prices found is', np.mean(solutions))