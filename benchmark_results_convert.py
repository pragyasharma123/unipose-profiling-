import pandas as pd
import json

# Load JSON data from a file
with open('benchmark_results.json', 'r') as file:
    data = json.load(file)

# Convert the 'gpu_records' list to a DataFrame
gpu_usage_df = pd.DataFrame(data['gpu_records'])

# Calculate the average power
average_power = gpu_usage_df['power'].mean()

print("Average Power:", average_power)
