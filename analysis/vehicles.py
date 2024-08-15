import json
import pandas as pd

from analysis.utils import check_instances


def vehicle_count(routes):
    output = 0
    for route in routes:
        if len(route) > 0:
            output += 1
    return output


def avg_vehicle_count(folder):
    count = []
    for key in folder:
        count.append(vehicle_count(folder[key]))
    return sum(count) / len(count)


def all_vehicle_counts(experiment, validated=True):
    if validated and experiment != 'c':
        instance_count = pd.read_csv("results/other/validate_count.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count.csv")

    if experiment == "a":
        include = instance_count[["A", "B", "E", "F", "M", "P", "CMT"]]
        include = include.drop(index=0, axis=0)
    elif experiment == "b":
        include = instance_count.drop(
            ["A", "B", "E", "F", "M", "P", "CMT", "id", "notes"], axis=1
        )
        include = include.drop(index=0, axis=0)
    elif experiment == "c":
        include = instance_count.drop(["id", "notes"], axis=1)

    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    for index, row in include.iterrows():
        print(row["id"])
        if experiment == "c":
            json_path = f"results/exp_{row['id']}/routes.json"
        else:
            json_path = f"results/exp_{row['id']}/routes_{experiment}.json"
        try:
            with open(json_path) as json_data:
                data = json.load(json_data)
            if pd.isna(row["notes"]):
                for key in data:
                    if row[key] == 1:
                        include.loc[index, key] = avg_vehicle_count(data[key])
            elif row["notes"] in ["greedy", "beam"]:
                for key in data:
                    if row[key] == 1:
                        include.loc[index, key] = avg_vehicle_count(
                            data[key][row["notes"]]
                        )

        except ValueError:
            # When none of the tests have been run
            pass

    include.to_csv(f"results/other/expt_{experiment}_vehicles.csv", index=False)
