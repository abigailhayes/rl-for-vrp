import json
import pandas as pd


def main():
    # Load in current counts
    settings_df = pd.read_csv('results/settings.csv')
    instances_df = pd.read_csv('results/instance_count.csv')


    # Update dataframe
    def instance_row(ident):
        output = {'id': ident}
        json_paths = [f'results/exp_{ident}/results_a.json', f'results/exp_{ident}/results_b.json']
        for json_path in json_paths:
            with open(json_path) as json_data:
                data = json.load(json_data)
            for key in data:
                output[key] = len(data[key])
        return pd.DataFrame.from_dict([output])


    # Run over new ids
    targets = [ident for ident in settings_df['id'] if ident not in instances_df['id']]
    for ident in targets:
        instances_df = pd.concat([instances_df, instance_row(ident)], ignore_index=True)

    # Save output
    instances_df.to_csv('results/instance_count.csv', index=False)

if __name__ == '__main__':
    main()
