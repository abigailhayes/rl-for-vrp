import json
import os
from instances.utils import import_instance
from itertools import pairwise


def get_cost(routes, instance):
    """Calculate the total cost of a solution to an instance"""
    costs = 0
    for r in routes:
        for i, j in pairwise([0] + r + [0]):
            costs += instance["edge_weight"][i][j]
    return costs

costs = {}
for test_set in ["A", "B", "E", "F", "M", "P", "CMT"]:
    costs[test_set] = {}
    for example in next(os.walk(f"instances/CVRP/{test_set}"))[2]:
        if example.endswith("sol"):
            continue
        example = example.replace(".vrp", "")
        data = import_instance(f"instances/CVRP/{test_set}", example)
        costs[test_set][f"{example}.vrp"] = get_cost(data['solution']['routes'], data['instance'])

with open(f"instances/expt_a_solns.json", "w") as f:
    json.dump(costs, f, indent=2)
