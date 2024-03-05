from itertools import pairwise

class VRPInstance:
    """A class for storing a VRP instance."""
    def __init__(self, instance):
        self.capacity = instance['capacity']
        self.demand = instance['demand']
        self.distance = instance['edge_weight']
        self.dimension = instance['dimension']
        self.routes = []

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