# Import

from methods.or_tools import ORtools
import methods.nazari.nazari as nazari
import methods.rl4co_run as rl4co_run
import utils

import os
import random
from datetime import date
import pandas as pd
import time


def main():
    start_time = time.time()
    args = utils.parse_experiment()

    # Looking at current ids in use as folders
    id_list = [int(str.replace(item, 'exp_', '')) for item in os.listdir('results') if 'exp' in item]

    # Determine ID of this run
    if len(id_list) == 0:
        ident = 1
    else:
        ident = max(id_list) + 1

    # Set up folder to save experiment results
    experiment_dir = f'results/exp_{str(ident)}'
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # set experiment seed
    random.seed(args['seed'])  # May need to look at more

    # Set up/train model (and save where appropriate)
    if args['method'] == 'nazari':
        model = nazari.Nazari(ident, task=args['method_settings']['task'])
        model.train_model()
        print("Finished training")
    elif args['method'] == 'rl4co':
        model = rl4co_run.RL4CO(args['problem'], args['method_settings']['init_method'], args['method_settings']['customers'], args['seed'], ident)
        model.set_model()
        model.train_model()
        print("Finished training")

    # Run tests
    if args['testing'] is not None:
        print("Testing...")
        if args['problem'] == 'CVRP':
            if args['method'] in ['nazari', 'rl4co']:
                utils.test_cvrp(args['method'], args['method_settings'], ident, args['testing'], model)
            elif args['method'] == 'ortools':
                utils.test_cvrp(args['method'], args['method_settings'], ident, args['testing'])
        elif args['problem'] == 'CVRPTW':
            if args['method'] == 'ortools':
                utils.test_cvrptw(args['method'], args['method_settings'], ident, args['testing'])

    end_time = time.time()
    # Create a dict with all variables of the current run
    settings = {**args, **args['method_settings']}
    settings.update({'id': ident, 'date': date.today(), 'time': end_time-start_time, 'testing': str(settings['testing'])})
    del settings['method_settings']
    print(settings)

    # Load dataframe that stores the results (every run adds a new row)
    settings_df = pd.read_csv('results/settings.csv')
    # Store settings in data frame
    settings_df = pd.concat([settings_df, pd.DataFrame.from_dict([settings])], ignore_index=True)
    # save updated csv file
    settings_df.to_csv('results/settings.csv', index=False)


if __name__ == '__main__':
    main()
