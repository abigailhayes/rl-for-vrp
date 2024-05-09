"""For carrying out comparisons relating to Experiment A - using established baselines"""

import json

import analysis.utils as analysis_utils


def compare_optimum(exp_filepath):
    """Compare a results file to the optimal baseline solutions."""
    # Load in data
    optima = analysis_utils.baseline_optima()
    with open(exp_filepath) as json_data:
        results = json.load(json_data)

    # Compare for each instance
    comparison = {}
    for test_set in results:
        comparison[test_set] = {}
        for instance in results[test_set]:
            try:
                if instance in optima[test_set] and results[test_set][instance] is dict:
                    # Compiled OR tools results format
                    comparison[test_set][instance] = (results[test_set][instance]['value'] - optima[test_set][
                        instance]) / optima[test_set][instance]
                elif instance in optima[test_set]:
                    # General results format
                    comparison[test_set][instance] = (results[test_set][instance] - optima[test_set][
                        instance]) / optima[test_set][instance]
                else:
                    # Nazari format
                    comparison[test_set][instance] = {}
                    comparison[test_set][instance]['greedy'] = (results[test_set]['greedy'][instance] - optima[
                        test_set][instance]) / optima[test_set][instance]
                    comparison[test_set][instance]['beam'] = (results[test_set]['beam'][instance] - optima[test_set][
                        instance]) / optima[test_set][instance]
            except NameError:
                pass

    return comparison
