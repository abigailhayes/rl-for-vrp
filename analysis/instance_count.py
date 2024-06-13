"""Counting the number of solutions returned in each CVRP experiment"""

import json
import pandas as pd


def main():
    # Load in current counts
    settings_df = pd.read_csv("results/settings.csv")
    instances_df = pd.read_csv("results/instance_count.csv")
    instances_df_tw = pd.read_csv("results/instance_count_tw.csv")
    # results[test][example]

    # Update dataframe
    def instance_row(ident):
        """Count the instances for a given id"""
        json_paths = [
            f"results/exp_{ident}/results_a.json",
            f"results/exp_{ident}/results_b.json",
        ]
        if (settings_df[settings_df["id"] == ident]["method"] == "nazari").item():
            # When data is split between beam and greedy
            output_greedy = {"id": ident, "notes": "greedy"}
            output_beam = {"id": ident, "notes": "beam"}
            for json_path in json_paths:
                with open(json_path) as json_data:
                    data = json.load(json_data)
                for key in data:
                    output_greedy[key] = len(data[key]["greedy"])
                    output_beam[key] = len(data[key]["beam"])
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
                        output[key] = len(data[key])
                except ValueError:
                    pass
            return pd.DataFrame.from_dict([output])

    def instance_row_tw(ident):
        """Count the instances for a given id for CVRPTW"""
        output = {"id": ident}
        try:
            with open(f"results/exp_{ident}/results.json") as json_data:
                data = json.load(json_data)
            for key in data:
                for variant in ['RC1', 'RC2', 'R1', 'R2', 'C1', 'C2']:
                    new_key = variant + "_" + str(key)
                    output[new_key] = sum([1 for instance in data[key].keys() if instance.startswith(variant)])
        except ValueError:
            pass
        return pd.DataFrame.from_dict([output])

    # Run over new ids for CVRP
    targets = [
        ident
        for ident in settings_df[settings_df["problem"] == "CVRP"]["id"]
        if ident not in list(instances_df["id"])
    ]

    for ident in targets:
        instances_df = pd.concat([instances_df, instance_row(ident)], ignore_index=True)

    # And for CVRPTW
    targets = [
        ident
        for ident in settings_df[settings_df["problem"] == "CVRPTW"]["id"]
        if ident not in list(instances_df["id"])
    ]

    for ident in targets:
        instances_df_tw = pd.concat(
            [instances_df_tw, instance_row_tw(ident)], ignore_index=True
        )

    # Save output
    instances_df.to_csv("results/instance_count.csv", index=False)
    instances_df_tw.to_csv("results/instance_count_tw.csv", index=False)


if __name__ == "__main__":
    main()
