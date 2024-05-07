"""For carrying out comparisons relating to Experiment A - using established baselines"""

import json

import analysis.utils as analysis_utils

def compare_optimum(exp_filepath):
    # Load in data
    optima = analysis_utils.baseline_optima()
    with open(exp_filepath) as json_data:
        results = json.load(json_data)

