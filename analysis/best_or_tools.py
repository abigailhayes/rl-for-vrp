import json
import pandas as pd


settings_df = pd.read_csv('results/settings.csv')
or_ids = settings_df[settings_df['method'] == 'ortools']['id'].tolist()

# Load in data
all_or = {}
for ident in or_ids:
    all_or[ident] = {}
    with open(f'results/exp_{ident}/results_a.json') as json_data:
        all_or['a'] = json.load(json_data)
    with open(f'results/exp_{ident}/results_b.json') as json_data:
        all_or['b'] = json.load(json_data)

# Find best solution

# Save result
