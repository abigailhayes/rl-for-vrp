from itertools import pairwise

def get_cost(instance, routes):
    # Calculate the total cost of a solution to an instance
    costs = 0
    for r in routes:
        pairs = list(pairwise([0]+r+[0]))
        for i,j in pairs:
            costs += instance['edge_weight'][i][j]
    return costs

def compare_cost(instance, solution, routes):
    route_cost = get_cost(instance, routes)
    return (route_cost-solution['cost'])/solution['cost']