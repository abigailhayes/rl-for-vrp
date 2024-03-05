from itertools import pairwise

class VRPInstance:
    """A class for storing a VRP instance."""
    def __init__(self, instance):
        self.capacity = instance['capacity']
        self.demand = instance['demand']
        self.distance = instance['edge_weight']
        self.dimension = instance['dimension']
        self.routes = []
        self.sol = False

    def get_cost(self):
        """Calculate the total cost of a solution to an instance"""
        costs = 0
        for r in self.routes:
            pairs = list(pairwise([0]+r+[0]))
            for i,j in pairs:
                costs += self.distance[i][j]
        self.cost = costs

    def add_sol(self, solution):
        """Add solution data for the instance"""
        self.sol = True
        self.sol_cost = solution['cost']
        self.sol_routes = solution['routes']

    def compare_cost(self):
        """Compare the current solution to the optimum"""
        self.get_cost()
        self.perc = (self.cost-self.sol_cost)/self.sol_cost