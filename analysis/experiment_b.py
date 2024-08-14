import json
import pandas as pd
import os

from analysis.utils import (
    add_settings,
    check_instances,
    average_distance,
    best_or_means,
    average_distance_multi,
    best_or_means_group_b,
)


def b_best():
    """Gather the best routes and their length for each example"""
    best_b = pd.read_csv("results/other/best_b.csv")["id"].to_list()
    settings_df = pd.read_csv("results/other/settings.csv")
    working_df = settings_df[settings_df["problem"] == "CVRP"]

    try:
        with open(f"results/other/optima_b.json") as json_data:
            optima_b = json.load(json_data)
        new = False
    except OSError:
        optima_b = {}
        new = True

    for ident in working_df["id"]:
        # Skip ids already checked
        if ident in best_b:
            continue
        # Load data
        if os.path.isfile(f"results/exp_{ident}/routes_b.json"):
            try:
                with open(f"results/exp_{ident}/results_b.json") as json_data:
                    results = json.load(json_data)
            except ValueError:
                continue
            try:
                with open(f"results/exp_{ident}/routes_b.json") as json_data:
                    routes = json.load(json_data)
            except ValueError:
                continue
        # Run through instances
        for subdir in next(os.walk("instances/CVRP/generate"))[1]:
            if new:
                optima_b[subdir] = {}
            for example in next(os.walk(f"instances/CVRP/generate/{subdir}"))[2]:
                if new:
                    optima_b[subdir][example] = {}
                try:
                    value = results[subdir].get(example)
                    if value is None:
                        continue
                    elif optima_b[subdir][example].get(
                        "id"
                    ) is None or value < optima_b[subdir][example].get("value"):
                        optima_b[subdir][example]["value"] = value
                        optima_b[subdir][example]["id"] = ident
                        optima_b[subdir][example]["route"] = routes[subdir][example]
                except KeyError:
                    pass
        best_b.append(ident)
    # Save result
    with open(f"results/other/optima_b.json", "w") as f:
        json.dump(optima_b, f, indent=2)
    df = pd.DataFrame(best_b, columns=["id"])
    df.to_csv("results/other/best_b.csv", index=False)


def best_b_means():
    # Load in relevant best b results
    json_path = f"results/other/optima_b.json"
    # When data is stored directly for each instance
    try:
        with open(json_path) as json_data:
            data = json.load(json_data)
    except ValueError:
        pass

    avgs = {"id": 0, "notes": "Experiment b best"}
    for key in data:
        avgs[key] = average_distance(
            {k: v["value"] for k, v in data[key].items() if len(v) > 0}
        )

    return pd.DataFrame.from_dict([avgs])


def b_all_averages(validated=True):
    """Get the averages for all experiment B instance types"""
    if validated:
        instance_count = pd.read_csv("results/other/validate_count.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count.csv")

    # Get a dataframe showing where averages should be taken
    include = instance_count.drop(
        ["A", "B", "E", "F", "M", "P", "CMT", "id", "notes"], axis=1
    )
    include = include.drop(index=0, axis=0)
    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    # Now go through and get averages
    for index, row in include.iterrows():
        print(row["id"])
        try:
            with open(f'results/exp_{row["id"]}/results_b.json') as json_data:
                data = json.load(json_data)
            if pd.isna(row["notes"]):
                for key in data:
                    if row[key] == 1:
                        include.loc[index, key] = average_distance(data[key])
            elif row["notes"] in ["greedy", "beam"]:
                for key in data:
                    if row[key] == 1:
                        include.loc[index, key] = average_distance(
                            data[key][row["notes"]]
                        )
        except ValueError:
            # When none of the Expt B tests have been run
            pass

    include = pd.concat(
        [include, best_or_means("b", instance_count), best_b_means()], ignore_index=True
    )

    include.to_csv("results/other/expt_b_means.csv", index=False)


def b_group_averages(validated=True):
    """Get the averages for all experiment B instance types"""
    if validated:
        instance_count = pd.read_csv("results/other/validate_count.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count.csv")

    # Get a dataframe showing where averages should be taken
    include = instance_count.drop(
        ["A", "B", "E", "F", "M", "P", "CMT", "id", "notes"], axis=1
    )
    include = include.drop(index=0, axis=0)
    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)

    # dictionary to define relations
    defns = {
        "cust_random": list(include.filter(regex="random_")),
        "cust_clustered": list(include.filter(regex="cluster_")),
        "depot_random": list(include.filter(regex="_random")),
        "depot_centre": list(include.filter(regex="_centre")),
        "depot_edge": list(include.filter(regex="_outer")),
        "cust_10": list(include.filter(regex=".*-10-\d+-\d+-\d+")),
        "cust_20": list(include.filter(regex=".*-20-\d+-\d+-\d+")),
        "cust_50": list(include.filter(regex=".*-50-\d+-\d+-\d+")),
        "cust_100": list(include.filter(regex=".*-100-\d+-\d+-\d+")),
        "demand_30": list(include.filter(regex=".*-\d+-30-\d+-\d+")),
        "demand_50": list(include.filter(regex=".*-\d+-50-\d+-\d+")),
        "demand_90": list(include.filter(regex=".*-\d+-90-\d+-\d+")),
    }

    include2 = instance_count[["id", "notes"]].copy()
    include2 = include2.drop(index=0, axis=0)
    # Need to sum over relevant columns
    for key, item in defns.items():
        include2[key] = include[item].sum(axis=1)

    # Now convert to binary
    include2.iloc[:, 2:] = include2.iloc[:, 2:].apply(
        lambda x: (x == x.max()).astype(int)
    )

    # Now go through and get averages
    for index, row in include2.iterrows():
        print(row["id"])
        try:
            with open(f'results/exp_{row["id"]}/results_b.json') as json_data:
                data = json.load(json_data)
            if pd.isna(row["notes"]):
                for key, item in defns.items():
                    if row[key] == 1:
                        include2.loc[index, key] = average_distance_multi(data, item)
        except ValueError:
            # When none of the Expt B tests have been run
            pass

    include2 = pd.concat(
        [include2, best_or_means_group_b(defns)], ignore_index=True
    )

    include2.to_csv("results/other/expt_b_group_means.csv", index=False)


def size_table(size):
    """Look at instance sets of a specific size"""
    # Read in data
    dist_means = pd.read_csv("results/other/expt_b_means.csv")
    # Select columns for the relevant size
    col_names = [
        col
        for col in [col for col in list(dist_means) if len(col.split("-")) > 1]
        if col.split("-")[1] == str(size)
    ]
    output = dist_means[col_names + ["id", "notes"]]

    # Filter to only the rows with results
    output = output[output[col_names].sum(axis=1) > 0]
    output = add_settings(output)

    # Save
    output.to_csv(f"results/other/expt_b_{size}.csv", index=False)


def main():
    b_best()
    b_all_averages()
    for size in [10, 20, 50, 100]:
        size_table(size)


if __name__ == "__main__":
    main()
