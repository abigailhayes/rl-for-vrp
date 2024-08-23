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


def avg_vehicle_count_tw(subdict, variant):
    temp_dict = {k: v for k, v in subdict.items() if k.startswith(variant)}
    output = avg_vehicle_count(temp_dict)
    return output


def avg_vehicle_count_bestor(expt):
    json_path = f"results/other/or_results_{expt}.json"
    with open(json_path) as json_data:
        data = json.load(json_data)

    output = {"id": 0}
    for folder in data:
        count = []
        try:
            for key in data[folder]:
                count.append(vehicle_count(data[folder][key]["route"]))
            output[folder] = sum(count) / len(count)
        except KeyError:
            output[folder] = 0

    return pd.DataFrame.from_dict([output])


def avg_vehicle_count_tw_bestor():
    json_path = f"results/other/or_results_c.json"
    with open(json_path) as json_data:
        data = json.load(json_data)

    output = {"id": 0}
    for key in data:
        for variant in ["RC1", "RC2", "R1", "R2", "C1", "C2"]:
            new_key = variant + "_" + str(key)
            try:
                temp_dict = {
                    k: v["route"] for k, v in data[key].items() if k.startswith(variant)
                }
                output[new_key] = avg_vehicle_count(temp_dict)
            except KeyError:
                output[new_key] = 0
    return pd.DataFrame.from_dict([output])


def all_vehicle_counts(experiment, validated=True):
    if validated:
        instance_count = pd.read_csv("results/other/validate_count.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count.csv")

    if experiment == "a":
        include = instance_count[["A", "B", "E", "F", "M", "P", "CMT"]]
    elif experiment == "b":
        include = instance_count.drop(
            ["A", "B", "E", "F", "M", "P", "CMT", "id", "notes"], axis=1
        )

    include = include.drop(index=0, axis=0)
    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    for index, row in include.iterrows():
        print(row["id"])
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

        except (ValueError, FileNotFoundError):
            # When none of the tests have been run
            pass

    include = pd.concat(
        [include, avg_vehicle_count_bestor(experiment)], ignore_index=True
    )

    include.to_csv(f"results/other/expt_{experiment}_vehicles.csv", index=False)


def all_vehicle_counts_c(validated=True):
    if validated:
        instance_count = pd.read_csv("results/other/validate_count_tw.csv")
    else:
        instance_count = pd.read_csv("results/other/instance_count_tw.csv")

    include = instance_count.drop(["id", "notes"], axis=1)
    for column_name in list(include):
        include[column_name] = check_instances(include, column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    for index, row in include.iterrows():
        print(row["id"])
        json_path = f"results/exp_{int(row['id'])}/routes.json"
        if int(row["id"]) == 0:
            continue
        try:
            with open(json_path) as json_data:
                data = json.load(json_data)
            for key in data:
                for variant in ["RC1", "RC2", "R1", "R2", "C1", "C2"]:
                    new_key = variant + "_" + str(key)
                    if row[new_key] == 1:
                        include.loc[index, new_key] = avg_vehicle_count_tw(
                            data[key], variant
                        )

        except (ValueError, FileNotFoundError):
            # When none of the tests have been run
            pass

    include = pd.concat([include, avg_vehicle_count_tw_bestor()], ignore_index=True)

    include.to_csv(f"results/other/expt_c_vehicles.csv", index=False)
