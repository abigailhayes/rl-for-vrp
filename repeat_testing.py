import argparse
import pandas as pd
import os

import methods.rl4co_run as rl4co_run


def parse_run():
    # To handle dictionary input
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    """Parse arguments for an experiment run"""
    parser = argparse.ArgumentParser(description="Experiment arguments")
    parser.add_argument("--id", help="Specify random seed")

    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args


def main():
    args = parse_run()
    settings_df = pd.read_csv("results/other/settings.csv").set_index('id').to_dict("index")
    settings = settings_df[int(args["id"])]

    # Load in model
    if settings["method"] == "rl4co":
        model = rl4co_run.RL4CO(
            settings["problem"],
            settings["init_method"],
            settings["customers"],
            settings["id"],
            args["id"],
        )
        model.set_model()
        root_path = f"results/exp_{args['id']}/lightning_logs"
        long_path = f"{root_path}/{os.listdir(root_path)[0]}/checkpoints"
        model.model.load_from_checkpoint(
            f"{long_path}/{os.listdir(long_path)[0]}"
        )

    if settings['problem']=='CVRP':
        print('working')


if __name__ == "__main__":
    main()