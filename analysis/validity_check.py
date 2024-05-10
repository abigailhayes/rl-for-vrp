"""Carries out validity checks for returned routes on all RL results"""

import os
import json
import pandas as pd

import analysis.utils as analysis_utils


def main():
    settings_df = pd.read_csv('results/settings.csv')
    rl_ids = settings_df[settings_df['method'] != 'ortools']['id'].tolist()

    for ident in rl_ids:
        if os.path.isfile(f'results/exp_{ident}/validity_a.json'):
            continue
        else:
            analysis_utils.validate_experiment(ident)


if __name__ == '__main__':
    main()