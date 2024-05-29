import json
import pandas as pd


def check_instances(table, col_name: str):
    """Boolean indication of whether the full instance set is evaluated, returned as float"""
    output = table[col_name] == max(table[col_name])
    return output * 1.0


def average_distance(folder_dict: dict):
    """Given a dictionary with entries for each instance, takes the mean of the values"""
    output = []
    for key in folder_dict:
        output.append(folder_dict[key])
    return sum(output) / len(output)


def b_all_averages():
    """Get the averages for all experiment B instance types"""
    instance_count = pd.read_csv("results/instance_count.csv")

    # Get a dataframe showing where averages should be taken
    include = instance_count.drop(
        ["A", "B", "E", "F", "M", "P", "CMT", "id", "notes"], axis=1
    )
    include = include.drop(index=0, axis=0)
    for column_name in list(include):
        include[column_name] = check_instances(column_name)
    include["id"] = instance_count["id"]
    include["notes"] = instance_count["notes"]

    # Now go through and get averages
    for index, row in include.iterrows():
        if row["id"] > 10:  # FOR TESTING
            continue
        with open(f'results/exp_{row["id"]}/results_b.json') as json_data:
            data = json.load(json_data)
        if pd.isna(row["notes"]):
            for key in data:
                if row[key] == 1:
                    include.loc[index, key] = average_distance(data[key])
        elif row["notes"] in ["greedy", "beam"]:
            for key in data:
                if row[key] == 1:
                    include.loc[index, key] = average_distance(data[key][row["notes"]])

    include.to_csv("results/expt_b_means.csv", index=False)


def main():
    b_all_averages()

if __name__ == '__main__':
    main()
