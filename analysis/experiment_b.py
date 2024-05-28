import json
import pandas as pd

settings_df = pd.read_csv('results/settings.csv')
instance_count = pd.read_csv('results/instance_count.csv')

col_name = 'cluster_centre-20-30-100-42'


def ids_for_col(col_name):
    """Gets just the ids with solutions for all instances"""
    ids = instance_count[instance_count['id'] != 0][instance_count[instance_count['id'] != 0][col_name] == max(instance_count[instance_count['id'] != 0][col_name])][['id', 'notes']]
    return ids

def avg_for_col(col_name):
    ids = ids_for_col(col_name)

    for _, row in ids.iterrows():
        if pd.isna(row['notes']):
            # Load in certain way
        elif row['notes'] in ['greedy', 'beam']:
            # Load in another way

