import json
import pandas as pd

# TODO - ensure limited to CVRP

settings_df = pd.read_csv("results/settings.csv")
instance_count = pd.read_csv("results/instance_count.csv")


def check_instances(col_name):
    """Boolean indication of whether the full instance set is evaluated"""
    output = include[col_name] == max(include[col_name])
    return output


def average_distance(folder_dict):
    output = []
    for key in folder_dict:
        output.append(folder_dict[key])
    return sum(output)/len(output)


include = instance_count.drop(["id", "notes"], axis=1)
include = include.drop(index=0, axis=0)
for column_name in list(include):
    include[column_name] = check_instances(column_name)
include["id"] = instance_count["id"]
include["notes"] = instance_count["notes"]
for index, row in include.iterrows():
    if row["id"] > 10:  # FOR TESTING
        continue
    with open(f'results/exp_{row["id"]}/results_b.json') as json_data:
        data = json.load(json_data)
    if pd.isna(row["notes"]):
        for key in data:
            if row[key]:
                include[key][index] = average_distance(data[key])
    elif row["notes"] in ["greedy", "beam"]:
        print(row)
