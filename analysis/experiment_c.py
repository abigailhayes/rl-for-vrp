"""For carrying out comparisons relating to Experiment A - using established baselines"""

import json
import pandas as pd

from analysis.utils import check_instances, average_distance_tw, best_or_means


def c_all_averages(validated=True):
    """Get the averages for all experiment C instance types"""
    if validated:  # No validation set up yet
        instance_count = pd.read_csv("results/other/validate_count_tw.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count_tw.csv")

    # Get a dataframe showing where averages should be taken
    include = instance_count.drop(["id", "notes"], axis=1)
    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    # Now go through and get averages
    for index, row in include.iterrows():
        print(int(row["id"]))
        if int(row["id"]) == 0:
            continue
        try:
            with open(f'results/exp_{int(row["id"])}/results.json') as json_data:
                data = json.load(json_data)
            for key in data:
                for variant in ["RC1", "RC2", "R1", "R2", "C1", "C2"]:
                    new_key = variant + "_" + str(key)
                    if row[new_key] == 1:
                        include.loc[index, new_key] = average_distance_tw(
                            data[key], variant
                        )
        except ValueError:
            # When none of the Expt C tests have been run
            pass

    include = pd.concat(
            [include, best_or_means('c', instance_count)], ignore_index=True
        )

    include.to_csv("results/other/expt_c_means.csv", index=False)


def main():
    c_all_averages()


if __name__ == "__main__":
    main()
