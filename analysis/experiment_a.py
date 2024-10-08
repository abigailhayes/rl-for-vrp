"""For carrying out comparisons relating to Experiment A - using established baselines"""

import json
import pandas as pd

from statistics import mean

from analysis.utils import check_instances, best_or_means


def a_compare_optimum(exp_filepath):
    """Compare a results file to the optimal baseline solutions."""
    # Load in data
    with open("instances/expt_a_solns.json") as json_data:
        optima = json.load(json_data)
    with open(exp_filepath) as json_data:
        results = json.load(json_data)

    # Compare for each instance
    comparison = {}
    for test_set in results:
        comparison[test_set] = {}
        for instance in results[test_set]:
            try:
                if instance in optima[test_set] and results[test_set][instance] is dict:
                    # Compiled OR tools results format
                    comparison[test_set][instance] = (
                        results[test_set][instance]["value"]
                        - optima[test_set][instance]
                    ) / optima[test_set][instance]
                elif instance in optima[test_set]:
                    # General results format
                    comparison[test_set][instance] = (
                        results[test_set][instance] - optima[test_set][instance]
                    ) / optima[test_set][instance]
                else:
                    # Nazari format
                    comparison[test_set][instance] = {}
                    comparison[test_set][instance]["greedy"] = (
                        results[test_set]["greedy"][instance]
                        - optima[test_set][instance]
                    ) / optima[test_set][instance]
                    comparison[test_set][instance]["beam"] = (
                        results[test_set]["beam"][instance] - optima[test_set][instance]
                    ) / optima[test_set][instance]
            except NameError:
                pass

    return comparison


def a_avg_compare(compare_dict, test_set):
    """Average the results for each instance set"""
    with open("instances/expt_a_solns.json") as json_data:
        optima = json.load(json_data)
    if test_set == "CMT":
        compare_dict = {
            key: compare_dict[key]
            for key in [
                "CMT1.vrp",
                "CMT2.vrp",
                "CMT3.vrp",
                "CMT4.vrp",
                "CMT5.vrp",
                "CMT11.vrp",
                "CMT12.vrp",
            ]
        }
    output = []
    for key in compare_dict:
        output.append(
            (compare_dict[key] - optima[test_set][key]) / optima[test_set][key]
        )
    return mean(output)


def a_all_averages(validated=True):
    """Get the averages for all experiment A instance types"""
    if validated == True:
        instance_count = pd.read_csv("results/other/validate_count.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count.csv")

    # Get a dataframe showing where averages should be taken
    include = instance_count[["A", "B", "E", "F", "M", "P", "CMT"]]
    include = include.drop(index=0, axis=0)
    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    # Now go through and get averages
    for index, row in include.iterrows():
        print(row["id"])
        try:
            with open(f'results/exp_{row["id"]}/results_a.json') as json_data:
                data = json.load(json_data)
            if pd.isna(row["notes"]):
                for key in data:
                    if row[key] == 1:
                        include.loc[index, key] = a_avg_compare(data[key], key)
            elif row["notes"] in ["greedy", "beam"]:
                for key in data:
                    if row[key] == 1:
                        include.loc[index, key] = a_avg_compare(
                            data[key][row["notes"]], key
                        )
        except ValueError:
            # When none of the Expt A tests have been run
            pass

    include = pd.concat(
        [include, best_or_means("a", instance_count)], ignore_index=True
    )

    include.to_csv("results/other/expt_a_means.csv", index=False)
