from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def plot_max_demand(size, cust_distn, depot_locatn):
    """A plot of each method, running over varying maximum demand for the problem"""

    raw_data = pd.read_csv(f"results/other/expt_b_{size}.csv").replace(0.0, np.NaN)

    plot_data = raw_data[
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
    ]
    linestyle_mapping = {
        style: dash_patterns[i % len(dash_patterns)]
        for i, style in enumerate(unique_styles)
    }

    fig, ax = plt.subplots()

    for name in set(plot_data["init_method"]):
        subset = plot_data[plot_data["init_method"] == name]
        line_style = linestyle_mapping[subset["method"].iloc[0]]
        (line,) = ax.plot(subset["max_demand"], subset["avg_dist"], label=name)
        if line_style[0] is not None:
            line.set_dashes(line_style)

    ax.set_xlabel("Maximum demand")
    ax.set_ylabel("Average distance")
    ax.legend(loc="best")

    plt.savefig(f"analysis/plots/md_{cust_distn}_{depot_locatn}_{size}.svg")


def plot_dstn_sets(size, max_demand):

    raw_data = pd.read_csv(f"results/other/expt_b_{size}.csv").replace(0.0, np.NaN)

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
    plot_data["prob_set"] = plot_data["prob_set"].str.split("-").str[0]

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

    plt.savefig(f"analysis/plots/ds_{max_demand}_{size}.svg")
