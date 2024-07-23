import argparse
import pandas as pd
import os
import ast
import json
from datetime import datetime

from lightning.pytorch import seed_everything

import methods.nazari.nazari as nazari
import methods.rl4co_run as rl4co_run
import utils


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
    parser.add_argument("--id", help="Specify id")
    parser.add_argument(
        "--seeds", default=[], help="Specify seeds", type=int, nargs="*"
    )

    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args


def main():
    args = parse_run()
    settings_df = (
        pd.read_csv("results/other/settings.csv").set_index("id").to_dict("index")
    )
    settings = settings_df[int(args["id"])]
    model = None

    # Load in model
    if settings["method"] == "rl4co":
        try:
            args["method_settings"]["decode"]
        except NameError:
            args["method_settings"]["decode"] = "greedy"

        model = rl4co_run.RL4CO(
            settings["problem"],
            settings["init_method"],
            settings["customers"],
            settings["seed"],
            args["id"],
        )
        model.set_model()
        root_path = f"results/exp_{args['id']}/lightning_logs"
        long_path = f"{root_path}/{os.listdir(root_path)[0]}/checkpoints"
        model.model.load_from_checkpoint(f"{long_path}/{os.listdir(long_path)[0]}")

    elif settings["method"] == "nazari":

        def get_earliest_date_option(options):
            # Define a function to extract the date and time from the option string
            def extract_datetime(option):
                # Split the string and take the part containing the date and time
                date_time_str = (
                    option.split("-")[1] + "-" + option.split("-")[2] + "-" + option.split("-")[3].split("_")[0] + " " + option.split("_")[2].replace("-", ":")
                )
                # Convert the date_time_str to a datetime object
                return datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

            # Find the option with the earliest date
            earliest_option = min(options, key=extract_datetime)
            return earliest_option

        model = nazari.Nazari(args["id"], settings["task"])
        root_path = f"results/exp_{args['id']}/logs"
        model.agent.args["load_path"] = f"{root_path}/{get_earliest_date_option(os.listdir(root_path))}/model"
        model.agent.load_model()

    results = {}
    for seed in args["seeds"]:
        results[seed] = {}
        seed_everything(seed, workers=True)
        if settings["problem"] == "CVRP":
            results[seed]["results"], results[seed]["routes"] = utils.test_cvrp(
                settings["method"],
                {
                    "init_method": settings["init_method"],
                    "improve_method": settings["improve_method"],
                },
                args["id"],
                ast.literal_eval(settings["testing"]),
                model,
                save=False,
            )

    with open(f"results/exp_{args['id']}/results_extra.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
