from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import json

from analysis.utils import average_distance


def plot_max_demand(size, cust_distn, depot_locatn, cust_train):
    """A plot of each method, running over varying maximum demand for the problem"""

    raw_data = pd.read_csv(f"results/other/expt_b_{size}.csv").replace(0.0, np.NaN)
    raw_data = raw_data[~raw_data["training"].isin(["old", "Old"])]
    raw_data.loc[raw_data["method"] == "rl4co_tsp", "init_method"] = "am tsp"
    raw_data.loc[raw_data["method"] == "rl4co_tsp", "method"] = "rl4co"

    plot_data = raw_data[
        (raw_data["customers"] == cust_train) | (raw_data["customers"].isna())
    ][raw_data["seed"] == 1][
        [col for col in raw_data if col.startswith(f"{cust_distn}_{depot_locatn}")]
        + ["method", "init_method"]
    ].melt(
        id_vars=["method", "init_method"], var_name="prob_set", value_name="avg_dist"
    )
    plot_data["max_demand"] = pd.to_numeric(plot_data["prob_set"].str.split("-").str[2])

    unique_styles = plot_data["method"].unique()
    dash_patterns = [
        (None, None),  # solid
        (3, 5),  # dashed
        (1, 5),  # dotted
        (3, 5, 1, 5),  # dashdot
        (5, 5),
    ]
    linestyle_mapping = {
        style: dash_patterns[i % len(dash_patterns)]
        for i, style in enumerate(unique_styles)
    }

    fig, ax = plt.subplots()

    for name in set(plot_data["init_method"]):
        subset = plot_data[plot_data["init_method"] == name]
        if subset.empty:
            print(f"No data for init_method: {name}")
            continue
        line_style = linestyle_mapping[subset["method"].iloc[0]]
        (line,) = ax.plot(subset["max_demand"], subset["avg_dist"], label=name)
        if line_style[0] is not None:
            line.set_dashes(line_style)

    ax.set_xlabel("Maximum demand")
    ax.set_ylabel("Average distance")
    ax.legend(loc="best")

    plt.savefig(
        f"analysis/plots/md_{cust_distn}_{depot_locatn}_{size}_{cust_train}.svg"
    )
    plt.close()


def plot_dstn_sets(size, max_demand):

    raw_data = pd.read_csv(f"results/other/expt_b_{size}.csv").replace(0.0, np.NaN)
    raw_data = raw_data[~raw_data["training"].isin(["old", "Old"])]

    plot_data = raw_data.copy()[
        [
            col
            for col in [
                col for col in raw_data if col.startswith(("random", "cluster"))
            ]
            if col.split("-")[2] == str(max_demand)
        ]
        + ["method", "init_method"]
    ].melt(
        id_vars=["method", "init_method"], var_name="prob_set", value_name="avg_dist"
    )
    plot_data["prob_set"] = (
        plot_data["prob_set"].str.split("-").str[0].str.replace("_", "\n")
    )

    plot_data.dropna(subset=["avg_dist"], inplace=True)

    fig, ax = plt.subplots()

    unique_styles = plot_data["method"].unique()
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "X",
        "d",
        "|",
        "_",
    ]
    marker_mapping = {
        style: markers[i % len(markers)] for i, style in enumerate(unique_styles)
    }

    for name in set(plot_data["init_method"]):
        subset = plot_data[plot_data["init_method"] == name]
        if subset.empty:
            print(f"No data for init_method: {name}")
            continue
        style_key = subset["method"].iloc[0]
        marker = marker_mapping.get(style_key, "o")  # Default to 'o' if not found
        ax.scatter(
            subset["prob_set"],
            subset["avg_dist"],
            label=name,
            marker=marker,
            s=75,
            alpha=0.6,
        )

    ax.set_xlabel("Problem set")
    ax.set_ylabel("Average distance")
    ax.legend(loc="best")

    plt.subplots_adjust(bottom=0.2)

    plt.savefig(f"analysis/plots/ds_{max_demand}_{size}.svg")
    plt.close()


def plot_seed(variant):
    """Compares the variation when models are trained from different seeds"""
    raw_data = pd.read_csv(f"results/other/expt_b_10.csv").replace(0.0, np.NaN)
    raw_data["id"] = raw_data["id"].replace(np.nan, 0)
    raw_data["init_method"] = raw_data["init_method"].replace(np.nan, "best")
    raw_data.loc[raw_data["id"].isin([74, 77]), "notes"] = "Seed A"
    raw_data.loc[raw_data["id"].isin([78, 91]), "notes"] = "Seed B"
    raw_data.loc[raw_data["id"].isin([84, 92]), "notes"] = "Seed C"
    raw_data = raw_data[
        (raw_data["id"].isin([0, 74, 78, 84, 77, 91, 92]))
        & (~raw_data["notes"].isin(["OR tools best"]))
    ]

    use_data = raw_data.copy().melt(
        id_vars=["notes", "init_method"],
        value_vars=[i for i in list(raw_data) if i.startswith(variant)],
    )
    use_data = use_data.groupby(["init_method", "variable"], as_index=False).value.agg(
        ["mean", "min", "max"]
    )
    use_data["split"] = (
        use_data["variable"].str.split("-").str[0].str.replace(variant + "_", "")
        + "\n"
        + use_data["variable"].str.split("-").str[2]
        + " "
        + use_data["variable"].str.split("-").str[3]
    )

    fig, ax = plt.subplots()
    for name in set(use_data["init_method"]):
        subset = use_data[use_data["init_method"] == name]
        plt.errorbar(
            subset["split"],
            subset["mean"],
            yerr=[subset["mean"] - subset["min"], subset["max"] - subset["mean"]],
            fmt="x",
            label=name,
        )
    ax.legend(loc="best")
    ax.set_xlabel("Problem set")
    ax.set_ylabel("Average distance")

    plt.subplots_adjust(bottom=0.2)

    plt.savefig(f"analysis/plots/seed_{variant}.svg")
    plt.close()


def plot_epochs():
    """Looks at how average results change with more epochs of training"""
    with open(f"results/am_epochs/results_am_10.json") as json_data:
        data = json.load(json_data)

    data["20"] = {}
    with open(f"results/exp_74/results_a.json") as json_data:
        data["20"]["a"] = json.load(json_data)
    with open(f"results/exp_74/results_b.json") as json_data:
        data["20"]["b"] = json.load(json_data)

    averages_a = {}
    averages_b = {}
    for key in data:
        averages_a[key] = {}
        averages_b[key] = {}
        for key2 in data[key]["a"]:
            averages_a[key][key2] = average_distance(data[key]["a"][key2])
        for key2 in data[key]["b"]:
            averages_b[key][key2] = average_distance(data[key]["b"][key2])
    table_a = pd.DataFrame.from_dict(averages_a, orient="index")
    table_b = pd.DataFrame.from_dict(averages_b, orient="index")

    fig, ax = plt.subplots()
    for column in table_a.columns:
        ax.plot(table_a[column], label=column)
    ax.legend(loc="best")
    plt.title("Standard test sets")
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Average distance")

    plt.savefig(f"analysis/plots/epochs_a.svg")
    plt.close()

    for size in ["10", "20", "50", "100"]:
        fig, ax = plt.subplots()
        for column in [i for i in table_b if i.split("-")[1] == size]:
            ax.plot(table_b[column])
        plt.title(f"Generated test sets - {size} customers")
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Average distance")

        plt.savefig(f"analysis/plots/epochs_b_{size}.svg")
        plt.close()


def main():
    sizes = [10, 20, 50, 100]
    cust_distn = ["random", "cluster"]
    depot_locatn = ["centre", "random", "outer"]
    max_demand = [90, 50, 30]
    cust_train = [10, 25, 50]

    for size, cust, depot, train in product(
        *[sizes, cust_distn, depot_locatn, cust_train]
    ):
        plot_max_demand(size, cust, depot, train)

    for size, demand in product(*[sizes, max_demand]):
        plot_dstn_sets(size, demand)

    plot_seed("random")
    plot_seed("cluster")

    plot_epochs()


if __name__ == "__main__":
    main()
