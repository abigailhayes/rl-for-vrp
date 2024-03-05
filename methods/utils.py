from itertools import pairwise

class VRPInstance:
    """A class for storing a VRP instance."""
    def __init__(self, instance):
        self.capacity = instance['capacity']
        self.demand = instance['demand']
        self.distance = instance['edge_weight']
        self.dimension = instance['dimension']
        self.routes = []

    def get_cost(self):
        """Calculate the total cost of a solution to an instance"""
        costs = 0
        for r in self.routes:
            pairs = list(pairwise([0]+r+[0]))
            for i,j in pairs:
                costs += self.edge_weight[i][j]
        self.cost = costs

def compare_cost(instance, solution, routes):
    route_cost = get_cost(instance, routes)
    return (route_cost-solution['cost'])/solution['cost']