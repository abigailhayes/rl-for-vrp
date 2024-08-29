from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import json
from scipy import stats

from analysis.utils import average_distance, average_distance_tw


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
        f"analysis/plots/md_{cust_distn}_{depot_locatn}_{size}_{cust_train}.png"
    )
    plt.close()


def plot_dstn_sets(size, max_demand):

    raw_data = pd.read_csv(f"results/other/expt_b_{size}.csv").replace(
        0.0, np.NaN
    )  # Replace 0 with na
    raw_data = raw_data[~raw_data["training"].isin(["old", "Old"])]  # Old flawed runs
    raw_data = raw_data[
        ~raw_data["method"].isin(["rl4co_tsp"])
    ]  # Appear as duplicates, worse performing anyway
    raw_data = raw_data[
        raw_data["customers"].isin([size, np.NaN])
    ]  # Keep heuristics and models trained for the same size, for simplicity
    raw_data = raw_data[raw_data["seed"] == 1]  # Keep only the first run of each
    raw_data = raw_data[
        raw_data["init_method"] != "mdam"
    ]  # Much worse performing so makes graphs wrong scale

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

    plt.savefig(f"analysis/plots/ds_{max_demand}_{size}.png")
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

    plt.savefig(f"analysis/plots/seed_{variant}.png")
    plt.close()


def plot_epochs():
    """Looks at how average results change with more epochs of training"""
    paths = [
        f"results/am_epochs/results_am_10.json",
        f"results/am_epochs/results_am_10_tw.json",
    ]
    for path in paths:
        with open(path) as json_data:
            data = json.load(json_data)

        if path == f"results/am_epochs/results_am_10.json":
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

            plt.savefig(f"analysis/plots/epochs_a.png")
            plt.close()

            for size in ["10", "20", "50", "100"]:
                fig, ax = plt.subplots()
                for column in [i for i in table_b if i.split("-")[1] == size]:
                    ax.plot(table_b[column])
                plt.title(f"Generated test sets - {size} customers")
                ax.set_xlabel("Training epochs")
                ax.set_ylabel("Average distance")

                plt.savefig(f"analysis/plots/epochs_b_{size}.png")
                plt.close()

        else:
            with open(f"results/exp_46/results.json") as json_data:
                data["20"] = json.load(json_data)

            groups = ["RC1", "RC2", "R1", "R2", "C1", "C2"]
            averages_c = {}
            sizes = ["25", "50", "100"]
            for size in sizes:
                averages_c[size] = {}
                for key in data:
                    averages_c[size][key] = {}
                    for group in groups:
                        averages_c[size][key][group] = average_distance_tw(
                            data[key][size], group
                        )
                table_c = pd.DataFrame.from_dict(averages_c[size], orient="index")

                fig, ax = plt.subplots()
                for column in table_c.columns:
                    ax.plot(table_c[column], label=column)
                ax.legend(loc="best")
                plt.title("Standard test sets")
                ax.set_xlabel("Training epochs")
                ax.set_ylabel("Average distance")

                plt.savefig(f"analysis/plots/epochs_c_{size}.png")
                plt.close()


def data_b_sizes(ident):
    """Organise data for the related plot function"""
    # Read in all means, and keep relevant id
    raw_data = (
        pd.read_csv(f"results/other/expt_b_means.csv")
        .replace(0.0, np.NaN)
        .drop("notes", axis=1)
    )
    raw_data = raw_data.loc[~(raw_data.iloc[:, 1:] == 0).any(axis=1)].dropna()
    if ident > 0:
        raw_data = raw_data[raw_data["id"] == ident]

    # Flip so that each column is a row
    melted_df = raw_data.melt(id_vars=["id"], var_name="column", value_name="value")

    # Split up original column name
    extracted_df = melted_df["column"].str.extract(
        r"([^_]+)_([^_]+)-(\d+)-(\d+)-\d+-\d+"
    )
    extracted_df.columns = ["distn", "depot", "cust", "demand"]

    # Combine to dataframe ready for plotting
    plot_df = pd.concat([extracted_df, melted_df[["value"]]], axis=1)
    plot_df["cust"] = pd.to_numeric(plot_df["cust"])
    plot_df["demand"] = pd.to_numeric(plot_df["demand"])

    return plot_df


def plot_b_sizes(ident):
    """Looking at the influences on solution size"""
    plot_df = data_b_sizes(ident)
    # Plot all variations
    x_variables = ["cust", "demand"]
    colour_variables = ["distn", "depot"]
    for x_variable, colour_variable in product(*[x_variables, colour_variables]):
        # Set up colours
        unique_values = plot_df[colour_variable].unique()
        colours = plt.cm.get_cmap("viridis", len(unique_values)).colors
        colour_map = {value: colours[i] for i, value in enumerate(unique_values)}

        # Plot each category with a specific colour
        plt.figure(figsize=(10, 6))

        for value, color in colour_map.items():
            subset = plot_df[plot_df[colour_variable] == value]
            plt.scatter(
                x=subset[x_variable],
                y=subset["value"],
                color=color,
                label=value,
                s=100,
                alpha=0.75,
            )

        # Add labels and title
        if x_variable == "cust":
            plt.xticks([10, 20, 50, 100])
            plt.xlabel("Number of customers")
        elif x_variable == "demand":
            plt.xticks([30, 50, 90])
            plt.xlabel("Maximum customer demand")
        plt.ylabel("Average solution distance")

        if colour_variable == "distn":
            plt.legend(title="Customer distribution")
        elif colour_variable == "depot":
            plt.legend(title="Depot location")

        plt.savefig(f"analysis/plots/size_b_{ident}_{x_variable}_{colour_variable}.png")
        plt.close()


def stats_b_sizes():
    start_df = data_b_sizes(-1)

    start_df = pd.get_dummies(start_df, columns=["depot", "distn"], drop_first=True)

    # Define independent variables (features) and dependent variable (target)
    X = start_df.drop(columns=["value"])
    y = start_df["value"]

    X = X.astype(np.float64)
    y = y.astype(np.float64)

    # Add a constant (intercept) to the model
    X = np.column_stack((np.ones(X.shape[0]), X))

    variable_names = ["Intercept"] + list(start_df.drop(columns=["value"]).columns)

    # Compute regression coefficients
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    XtY = X.T @ y
    coefficients = XtX_inv @ XtY

    # Compute predicted values and residuals
    y_pred = X @ coefficients
    residuals = y - y_pred

    # Compute R-squared
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    SS_res = np.sum(residuals**2)
    r2 = 1 - (SS_res / SS_tot)

    print("Regression Coefficients:")
    for name, coef in zip(variable_names, coefficients):
        print(f"{name}: {coef:.4f}")

    print(f"R-squared: {r2:.4f}")


def distance_vs_vehicles(expt):
    if expt in ["a", "c"]:
        distances = pd.read_csv(f"analysis/tables/expt_{expt}_means.csv").drop(
            ["notes"], axis=1
        )
        vehicles = pd.read_csv(f"analysis/tables/expt_{expt}_vehicles.csv").drop(
            ["notes"], axis=1
        )

    distances_long = pd.melt(
        distances,
        id_vars=["id", "method", "init_method", "customers"],
        var_name="variable",
        value_name="distances",
    )
    vehicles_long = pd.melt(
        vehicles,
        id_vars=["id", "method", "init_method", "customers"],
        var_name="variable",
        value_name="vehicles",
    )

    merged_df = pd.merge(
        distances_long,
        vehicles_long,
        on=["id", "method", "init_method", "customers", "variable"],
    )
    merged_df["init_method"] = merged_df["init_method"].fillna("ortools")
    merged_df = merged_df[merged_df["vehicles"] != 0]
    merged_df = merged_df[merged_df["init_method"] != "mdam"]

    unique_methods = merged_df["init_method"].unique()
    colors = plt.cm.get_cmap("viridis", len(unique_methods)).colors
    color_map = {method: colors[i] for i, method in enumerate(unique_methods)}

    plt.figure(figsize=(10, 6))
    for method, color in color_map.items():
        subset = merged_df[merged_df["init_method"] == method]
        plt.scatter(
            x=subset["distances"],
            y=subset["vehicles"],
            color=color,
            label=method,
            s=100,
            alpha=0.75,
        )
    if expt != "a":
        plt.title(
            f"Average distances vs Average number of vehicles for Experiment {expt.upper()}"
        )
        plt.xlabel("Average distances")
    else:
        plt.title(
            f"Average proportion worse than optima vs Average number of vehicles for Experiment {expt.upper()}"
        )
        plt.xlabel("Average proportion")
    plt.ylabel("Average number of vehicles")
    plt.legend(title="Method")
    plt.savefig(f"analysis/plots/distance_vs_vehicles_{expt}.png")
    plt.close()


def expt_c_sizes():

    df = (
        pd.read_csv(f"analysis/tables/expt_c_means.csv")
        .drop(["notes"], axis=1)
        .drop(index=0)
        .reset_index(drop=True)
    )
    df["init_method"] = df["init_method"].fillna("OR tools")

    df_long = pd.melt(
        df,
        id_vars=["id", "method", "init_method", "customers"],
        var_name="Metric",
        value_name="Value",
    )
    df_long = df_long[df_long["Value"] != 0].reset_index(drop=True)

    # Extract the base metric name (RC1, RC2, etc.) and the size (25, 50, 100)
    df_long[["Metric", "Size"]] = df_long["Metric"].str.extract(r"([A-Z]+[0-9]*)_(\d+)")

    # Convert the Size to numeric
    df_long["Size"] = df_long["Size"].astype(int)

    # Now, plot each metric separately using plain Matplotlib
    metrics = df_long["Metric"].unique()
    init_methods = df_long["init_method"].unique()
    color_map = {method: plt.cm.tab10(i) for i, method in enumerate(init_methods)}

    # Loop through each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Loop through each unique id
        for ident in df_long["id"].unique():
            # Filter data for the current metric and id
            model_data = df_long[
                (df_long["Metric"] == metric) & (df_long["id"] == ident)
            ]

            # Get the color based on init_method
            color = color_map[model_data["init_method"].iloc[0]]

            # Plot with shared color based on init_method
            plt.plot(
                model_data["Size"],
                model_data["Value"],
                label=model_data["init_method"].iloc[0],
                color=color,
            )

        # Create a custom legend with unique init_method values
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="init_method")

        plt.title(f"{metric} vs Size")
        plt.xlabel("Size")
        plt.ylabel(f"{metric} Value")
        plt.savefig(f"analysis/plots/expt_c_{metric}.png")
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

    distance_vs_vehicles("a")
    distance_vs_vehicles("c")

    expt_c_sizes()


if __name__ == "__main__":
    main()
