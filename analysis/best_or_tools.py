"""Combining the CVRP results from the OR tools methods"""

import os
import json
import vrplib

import pandas as pd

import analysis.utils as analysis_utils
import instances.utils as instance_utils


def main():
    # Read in settings file to get OR tools ids
    settings_df = pd.read_csv("results/other/settings.csv")
    or_ids = settings_df[settings_df["method"] == "ortools"][
        settings_df["problem"] == "CVRP"
    ]["id"].tolist()
    or_ids_tw = settings_df[settings_df["method"] == "ortools"][
        settings_df["problem"] == "CVRPTW"
    ]["id"].tolist()

    # Load in results for all OR tools CVRP experiments
    all_or = {}
    routes_or = {}
    for ident in or_ids:
        all_or[ident] = {}
        routes_or[ident] = {}
        try:
            with open(f"results/exp_{ident}/results_a.json") as json_data:
                all_or[ident]["a"] = json.load(json_data)
        except ValueError:
            pass
        try:
            with open(f"results/exp_{ident}/routes_a.json") as json_data:
                routes_or[ident]["a"] = json.load(json_data)
        except ValueError:
            pass
        if os.path.isfile(f"results/exp_{ident}/routes_b.json"):
            try:
                with open(f"results/exp_{ident}/results_b.json") as json_data:
                    all_or[ident]["b"] = json.load(json_data)
            except ValueError:
                pass
            try:
                with open(f"results/exp_{ident}/routes_b.json") as json_data:
                    routes_or[ident]["b"] = json.load(json_data)
            except ValueError:
                pass
    # Same for CVRPTW
    all_or_tw = {}
    routes_or_tw = {}
    for ident in or_ids_tw:
        all_or_tw[ident] = {}
        routes_or_tw[ident] = {}
        try:
            with open(f"results/exp_{ident}/results.json") as json_data:
                all_or_tw[ident] = json.load(json_data)
        except ValueError:
            pass
        try:
            with open(f"results/exp_{ident}/routes.json") as json_data:
                routes_or_tw[ident] = json.load(json_data)
        except ValueError:
            pass

    # Find best solution
    # Test set A
    test_sets = ["A", "B", "E", "F", "M", "P", "CMT"]
    output_a = {}
    for test_set in test_sets:
        output_a[test_set] = {}
        for example in next(os.walk(f"instances/CVRP/{test_set}"))[2]:
            if example.endswith("sol"):
                continue
            output_a[test_set][example] = {}
            data = vrplib.read_instance(f"instances/CVRP/{test_set}/{example}")
            for ident in or_ids:
                value = all_or[ident]["a"][test_set].get(example)
                if value is None:
                    continue
                elif (
                    output_a[test_set][example].get("id") is None
                    or value < output_a[test_set][example].get("value")
                ) and analysis_utils.validate_routes(
                    routes_or[ident]["a"][test_set][example],
                    data["demand"],
                    data["capacity"],
                ) == 1:
                    output_a[test_set][example]["value"] = value
                    output_a[test_set][example]["id"] = ident
                    output_a[test_set][example]["route"] = routes_or[ident]["a"][
                        test_set
                    ][example]

    # Test set B
    output_b = {}
    for subdir in next(os.walk("instances/CVRP/generate"))[1]:
        output_b[subdir] = {}
        for example in next(os.walk(f"instances/CVRP/generate/{subdir}"))[2]:
            output_b[subdir][example] = {}
            data = vrplib.read_instance(f"instances/CVRP/generate/{subdir}/{example}")
            for ident in or_ids:
                try:
                    value = all_or[ident]["b"][subdir].get(example)
                    if value is None:
                        continue
                    elif (
                        output_b[subdir][example].get("id") is None
                        or value < output_b[subdir][example].get("value")
                    ) and analysis_utils.validate_routes(
                        routes_or[ident]["b"][subdir][example],
                        data["demand"],
                        data["capacity"],
                    ) == 1:
                        output_b[subdir][example]["value"] = value
                        output_b[subdir][example]["id"] = ident
                        output_b[subdir][example]["route"] = routes_or[ident]["b"][
                            subdir
                        ][example]
                except KeyError:
                    pass
    # Test set C
    output_c = {}
    for tester in ["25", "50", "100"]:
        output_c[tester] = {}
        for example in next(os.walk(f"instances/CVRPTW/Solomon"))[2]:
            if example.endswith("sol"):
                continue
            data = instance_utils.shrink_twinstance(
                vrplib.read_instance(
                    f"instances/CVRPTW/Solomon/{example}", instance_format="solomon"
                ),
                tester,
            )
            output_c[tester][example] = {}
            for ident in or_ids_tw:
                try:
                    value = all_or_tw[ident][tester].get(example)
                    if value is None:
                        continue
                    elif (
                        output_c[tester][example].get("id") is None
                        or value < output_c[tester][example].get("value")
                        and analysis_utils.validate_routes_tw(
                            routes_or_tw[ident][tester][example],
                            data,
                        )
                        == 1
                    ):
                        output_c[tester][example]["value"] = value
                        output_c[tester][example]["id"] = ident
                        output_c[tester][example]["route"] = routes_or_tw[ident][
                            tester
                        ][example]
                except KeyError:
                    pass

    # Save result
    with open(f"results/other/or_results_a.json", "w") as f:
        json.dump(output_a, f, indent=2)
    with open(f"results/other/or_results_b.json", "w") as f:
        json.dump(output_b, f, indent=2)
    with open(f"results/other/or_results_c.json", "w") as f:
        json.dump(output_c, f, indent=2)


if __name__ == "__main__":
    main()
