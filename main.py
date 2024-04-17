# Import

from methods.or_tools import ORtools
import utils

import os
import random
from datetime import date
import pandas as pd


def main():
    args = utils.parse_experiment()

    # Looking at current ids in use as folders
    id_list = [int(str.replace(item, 'exp_', '')) for item in os.listdir('results') if 'run' in item]

    # Determine ID of this run
    if len(id_list) == 0:
        ident = 1
    else:
        ident = max(id_list) + 1

    # Set up folder to save experiment results
    experiment_dir = f'results/exp_{str(id)}'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # set experiment seed
    random.seed(args['seed'])  # May need to look at more

    # Set up/train model (and save where appropriate)

    # Run tests

    # Create a dict with all variables of the current run
    args.update({'ID': ident, 'date': date.today()})
    print(args)

    # Load dataframe that stores the results (every run adds a new row)
    results_df = pd.read_csv('results/results.csv')
    # Store everything in data frame
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(args)], ignore_index=True)
    # save updated csv file
    results_df.to_csv('results/results.csv', index=False)


if __name__ == '__main__':
    main()
