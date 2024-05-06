import os
import json

import pandas as pd


settings_df = pd.read_csv('results/settings.csv')
or_ids = settings_df[settings_df['method'] == 'ortools']['id'].tolist()

# Load in data
all_or = {}
routes_or = {}
for ident in or_ids:
    all_or[ident] = {}
    routes_or[ident] = {}
    with open(f'results/exp_{ident}/results_a.json') as json_data:
        all_or[ident]['a'] = json.load(json_data)
    with open(f'results/exp_{ident}/results_b.json') as json_data:
        all_or[ident]['b'] = json.load(json_data)
    with open(f'results/exp_{ident}/results_a.json') as json_data:
        routes_or[ident]['a'] = json.load(json_data)
    with open(f'results/exp_{ident}/results_b.json') as json_data:
        routes_or[ident]['b'] = json.load(json_data)

# Find best solution
# Test set A
test_sets = ['A', 'B', 'E', 'F', 'M', 'P', 'CMT']
output_a = {}
for test_set in test_sets:
    output_a[test_set] = {}
    for example in next(os.walk(f'instances/CVRP/{test_set}'))[2]:
        if example.endswith('sol'):
            continue
        output_a[test_set][example] = {}
        for ident in or_ids:
            value = all_or[ident]['a'][test_set].get(example)
            if value is None:
                continue
            elif output_a[test_set][example].get('id') is None or value < output_a[test_set][example].get('value'):
                output_a[test_set][example]['value'] = value
                output_a[test_set][example]['id'] = ident
# Test set B
output_b = {}
for subdir in next(os.walk('instances/CVRP/generate'))[1]:
    output_b[subdir] = {}
    for example in next(os.walk(f'instances/CVRP/generate/{subdir}'))[2]:
        output_b[subdir][example] = {}
        for ident in or_ids:
            value = all_or[ident]['b'][subdir].get(example)
            if value is None:
                continue
            elif output_b[subdir][example].get('id') is None or value < output_b[subdir][example].get('value'):
                output_b[subdir][example]['value'] = value
                output_b[subdir][example]['id'] = ident

# Save result
