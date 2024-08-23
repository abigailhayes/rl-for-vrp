"""Carries out validity checks for returned routes on all RL results"""

import json
import pandas as pd

from analysis.utils import validate_dict


def validate_row(settings_df, ident):
    """Count the valid routes for a given id"""
    json_paths = [
        f"results/exp_{ident}/routes_a.json",
        f"results/exp_{ident}/routes_b.json",
    ]
    if (settings_df[settings_df["id"] == ident]["method"] == "nazari").item():
        # When data is split between beam and greedy
        output_greedy = {"id": ident, "notes": "greedy"}
        output_beam = {"id": ident, "notes": "beam"}
        for json_path in json_paths:
            with open(json_path) as json_data:
                data = json.load(json_data)
            for key in data:
                output_greedy[key] = validate_dict(data[key]["greedy"], key)
                output_beam[key] = validate_dict(data[key]["beam"], key)
        return pd.concat(
            [
                pd.DataFrame.from_dict([output_greedy]),
                pd.DataFrame.from_dict([output_beam]),
            ],
            ignore_index=True,
        )
    else:
        # When data is stored directly for each instance
        output = {"id": ident}
        for json_path in json_paths:
            try:
                with open(json_path) as json_data:
                    data = json.load(json_data)
                for key in data:
                    output[key] = validate_dict(data[key], key)
            except (ValueError, OSError):
                pass
        return pd.DataFrame.from_dict([output])


def validate_cvrp():
    # Load in current counts
    settings_df = pd.read_csv("results/other/settings.csv")
    valid_df = pd.read_csv("results/other/validate_count.csv")

    # Run over new ids
    targets = [
        ident
        for ident in settings_df[settings_df["problem"] == "CVRP"]["id"]
        if ident not in list(valid_df["id"])
    ]

    for ident in targets:
        valid_df = pd.concat([valid_df, validate_row(settings_df, ident)], ignore_index=True)

    # Save output
    valid_df.to_csv("results/other/validate_count.csv", index=False)


def main():
    validate_cvrp()


if __name__ == "__main__":
    main()
