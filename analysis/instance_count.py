import json
import pandas as pd

ident = 1

# Load in current counts
instances_df = pd.read_csv('results/instance_count.csv')

# Update dataframe
output = {}
json_paths = [f'results/exp_{ident}/results_a.json', f'results/exp_{ident}/results_b.json']
for json_path in json_paths:
    with open(json_path) as json_data:
        data = json.load(json_data)
    for key in data:
        output[key] = len(data[key])

# Save output
instances_df.to_csv('results/instance_count.csv', index=False)