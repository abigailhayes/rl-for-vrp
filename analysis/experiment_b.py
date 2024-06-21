import json
import pandas as pd

from analysis.utils import add_settings, check_instances, average_distance


def b_all_averages(validated=True):
    """Get the averages for all experiment B instance types"""
    if validated == True:
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

    include.to_csv("results/other/expt_b_means.csv", index=False)


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
    b_all_averages()
    for size in [10, 20, 50, 100]:
        size_table(size)


if __name__ == "__main__":
    main()
