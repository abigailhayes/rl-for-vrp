import json
import pandas as pd


def vehicle_count(routes):
    output = 0
    for route in routes:
        if len(route) > 0:
            output += 1
    return output


def avg_vehicle_count(folder, nazari=False):
    if nazari:
        count_greedy = []
        count_beam = []
        for key in folder["greedy"]:
            count_greedy.append(vehicle_count(folder["greedy"][key]))
            count_beam.append(vehicle_count(folder["beam"][key]))
        return (sum(count_greedy) / len(count_greedy)), (
            sum(count_beam) / len(count_beam)
        )
    else:
        count = []
        for key in folder:
            count.append(vehicle_count(folder[key]))
        return sum(count) / len(count)



def add_vehicle_count(ident, experiment):
    settings_df = pd.read_csv("results/other/settings.csv")

    if experiment == "c":
        json_path = f"results/exp_{ident}/routes.json"
    else:
        json_path = f"results/exp_{ident}/routes_{experiment}.json"

    if (settings_df[settings_df["id"] == ident]["method"] == "nazari").item():
        with open(json_path) as json_data:
            data = json.load(json_data)
        # When data is split between beam and greedy
        output_greedy = {"id": ident, "notes": "greedy"}
        output_beam = {"id": ident, "notes": "beam"}

        for key in data:
            output_greedy[key], output_beam[key] = avg_vehicle_count(
                data[key], nazari=True
            )
        return pd.concat(
            [
                pd.DataFrame.from_dict([output_greedy]),
                pd.DataFrame.from_dict([output_beam]),
            ],
            ignore_index=True,
        )

    else:
        with open(json_path) as json_data:
            data = json.load(json_data)
        output = {"id": ident}
        for key in data:
            output[key] = avg_vehicle_count(data[key])
        return pd.DataFrame.from_dict([output])
