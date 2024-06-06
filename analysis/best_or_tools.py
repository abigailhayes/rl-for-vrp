"""Combining the CVRP results from the OR tools methods"""
import os
import json

import pandas as pd


def main():
    # Read in settings file to get OR tools ids
    settings_df = pd.read_csv('results/settings.csv')
    or_ids = settings_df[settings_df['method'] == 'ortools'][settings_df['problem'] == 'CVRP']['id'].tolist()

    # Load in results for all OR tools experiments
    all_or = {}
    routes_or = {}
    for ident in or_ids:
        all_or[ident] = {}
        routes_or[ident] = {}
        try:
            with open(f'results/exp_{ident}/results_a.json') as json_data:
                all_or[ident]['a'] = json.load(json_data)
        except ValueError:
            pass
        try:
            with open(f'results/exp_{ident}/routes_a.json') as json_data:
                routes_or[ident]['a'] = json.load(json_data)
        except ValueError:
            pass
        if os.path.isfile(f'results/exp_{ident}/routes_b.json'):
            try:
                with open(f'results/exp_{ident}/results_b.json') as json_data:
                    all_or[ident]['b'] = json.load(json_data)
            except ValueError:
                pass
            try:
                with open(f'results/exp_{ident}/routes_b.json') as json_data:
                    routes_or[ident]['b'] = json.load(json_data)
            except ValueError:
                pass

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
                    output_a[test_set][example]['route'] = routes_or[ident]['a'][test_set][example]

    # Test set B
    output_b = {}
    for subdir in next(os.walk('instances/CVRP/generate'))[1]:
        output_b[subdir] = {}
        for example in next(os.walk(f'instances/CVRP/generate/{subdir}'))[2]:
            output_b[subdir][example] = {}
            for ident in or_ids:
                try:
                    value = all_or[ident]['b'][subdir].get(example)
                    if value is None:
                        continue
                    elif output_b[subdir][example].get('id') is None or value < output_b[subdir][example].get('value'):
                        output_b[subdir][example]['value'] = value
                        output_b[subdir][example]['id'] = ident
                        output_b[subdir][example]['route'] = routes_or[ident]['b'][subdir][example]
                except KeyError:
                    pass

    # Save result
    with open(f'results/or_results_a.json', 'w') as f:
        json.dump(output_a, f, indent=2)
    with open(f'results/or_results_b.json', 'w') as f:
        json.dump(output_b, f, indent=2)

if __name__ == '__main__':
    main()
