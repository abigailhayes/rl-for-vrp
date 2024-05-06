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
test_sets = ['A', 'B', 'E', 'F', 'M', 'P', 'CMT']
output = {}
for test_set in test_sets:
    output[test_set] = {}
    for example in next(os.walk(f'instances/CVRP/{test_set}'))[2]:
        if example.endswith('sol'):
            continue
        output[test_set][example] = {}
        for ident in or_ids:
            value = all_or[ident]['a'][test_set].get(example)
            if value is None:
                continue
            elif output[test_set][example].get('id') == None or value < output[test_set][example].get('value'):
                output[test_set][example]['value'] = value
                output[test_set][example]['id'] = ident

            # Save result
