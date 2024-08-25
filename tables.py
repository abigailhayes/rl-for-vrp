import pandas as pd


def make_table(filename, problem):
    if problem == "cvrp":
        items = [0, 74, 100, 90, 76, 101, 99, 77, 102, 97, 80, 103, 81, 133, 134]
    elif problem == "cvrptw":
        items = [0, 46, 49, 108, 83, 65, 89, 67]

    data = pd.merge(
        pd.read_csv("results/other/settings.csv")[
            ["id", "method", "init_method", "customers"]
        ],
        pd.read_csv(f"results/other/{filename}.csv"),
        on="id",
        how="outer",
    )
    data = data[data["id"].isin(items)]
    data["id"] = pd.Categorical(data["id"], categories=items, ordered=True)
    data = data.sort_values("id")

    data.to_csv(f"analysis/tables/{filename}.csv", index=False)


def make_tables():
    make_table("validate_count", "cvrp")
    make_table("validate_count_tw", "cvrptw")

    make_table("expt_b_group_means_10", "cvrp")
    make_table("expt_b_group_means_20", "cvrp")
    make_table("expt_b_group_means_50", "cvrp")
    make_table("expt_b_group_means_100", "cvrp")
    make_table("expt_a_means", "cvrp")
    make_table("expt_c_means", "cvrptw")

    make_table("expt_a_vehicles", "cvrp")
    make_table("expt_c_vehicles", "cvrptw")
