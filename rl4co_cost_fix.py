import json
import vrplib
from itertools import pairwise


def get_cost(routes, instance):
    """Calculate the total cost of a solution to an instance"""
    costs = 0
    for r in routes:
        for i, j in pairwise([0] + r + [0]):
            costs += instance["edge_weight"][i][j]
    return costs


def cost_fix(ident):
    """Run a fix to the costs calculated for RL4CO implemented versions"""
    for expt in ["a", "b"]:
        with open(f"results/exp_{ident}/routes_{expt}.json") as json_data:
            routes = json.load(json_data)
        results = {}
        for key in routes:
            results[key] = {}
            for instance in routes[key]:
                if expt == "a":
                    data = vrplib.read_instance(f"instances/CVRP/{key}/{instance}")
                else:
                    data = vrplib.read_instance(
                        f"instances/CVRP/generate/{key}/{instance}"
                    )
                results[key][instance] = get_cost(routes[key][instance], data)
        with open(f"results/exp_{ident}/results_{expt}.json", "w") as f:
            json.dump(results, f, indent=2)
