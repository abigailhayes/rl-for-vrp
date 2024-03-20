from itertools import pairwise
import numpy as np

from methods.TSP.utils import TSPInstance


class VRPInstance:
    """A class for storing a VRP instance.
    - instance: need to provide an instance as input when creating"""
    def __init__(self, instance):
        self.capacity = instance['capacity']
        self.demand = instance['demand']
        self.distance = instance['edge_weight']
        self.dimension = instance['dimension']
        self.coords = instance['node_coord']
        self.routes = []
        self.sol = False

    def _cap_check(self, new_route):
        """Check that the new proposed route fits within the capacity demand"""
        return sum([self.demand[i] for i in new_route])

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

    def _gen_tsp_instance(self, cluster):
        """Takes a list of nodes and prepares an instance for giving to TSPInstance"""
        cluster = [0] + cluster
        instance = {}
        instance['cluster'] = cluster
        instance['dimension'] = len(cluster)
        instance['distance'] = self.distance[np.ix_(cluster, cluster)]
        instance['coords'] = self.coords[cluster]
        return TSPInstance(instance)

class NodePair:
    """Class for holding information about the current node pair"""

    def __init__(self, i, j, routes):
        self.i = i
        self.j = j
        self.routes = routes
        self.route_i, self.route_j = self._get_route(self.i), self._get_route(self.j)
        self.pos_i, self.pos_j = self._pos_check(self.route_i, self.i), self._pos_check(self.route_j, self.j)

    def _get_route(self, node):
        """Get the current route the nodes of interest are in"""
        return [r for r in self.routes if node in r][0]

    def _pos_check(self, route, node):
        """Check where in a route a node appears"""
        if route[0]==node:
            return 0
        elif route[-1]==node:
            return 1
        else:
            return 2
