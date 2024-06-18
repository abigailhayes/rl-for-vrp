from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


cvrp10_df = pd.read_csv("results/expt_b_10.csv").replace(0.0, np.NaN)

plot_data = cvrp10_df[
    [col for col in cvrp10_df if col.startswith("random_random")]
    + ["method", "init_method"]
].melt(id_vars=["method", "init_method"], var_name="prob_set", value_name="avg_dist")
plot_data["density"] = pd.to_numeric(plot_data["prob_set"].str.split("-").str[2])

unique_styles = plot_data['method'].unique()
dash_patterns = [
    (None, None),   # solid
    (3, 5),         # dashed
    (1, 5),         # dotted
    (3, 5, 1, 5)    # dashdot
]
linestyle_mapping = {style: dash_patterns[i % len(dash_patterns)] for i, style in enumerate(unique_styles)}

fig, ax = plt.subplots()

for name in set(plot_data["init_method"]):
    subset = plot_data[plot_data["init_method"] == name]
    line_style = linestyle_mapping[subset["method"].iloc[0]]
    (line,) = ax.plot(subset["density"], subset["avg_dist"], label=name)
    if line_style[0] is not None:
        line.set_dashes(line_style)

ax.set_xlabel("Density")
ax.set_ylabel("Average distance")
ax.legend(loc="best")


plot_data.set_index("density", inplace=True)
ax = plot_data.groupby("init_method")["avg_dist"].plot(x="density", y="avg_dist")
plt.savefig("analysis/plots/test.svg")
