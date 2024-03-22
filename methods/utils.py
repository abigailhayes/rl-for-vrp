from itertools import pairwise
import numpy as np

from methods.TSP.utils import TSPInstance


class VRPInstance:
    """A class for storing a VRP instance.
    - instance: need to provide an instance as input when creating"""
    def __init__(self, instance):
        self.perc = None
        self.sol_routes = None
        self.sol_cost = None
        self.cost = None
        self.polars = None
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

    def _get_cost(self, routes):
        """Calculate the total cost of a solution to an instance"""
        costs = 0
        for r in routes:
            for i, j in pairwise([0]+r+[0]):
                costs += self.distance[i][j]
        return costs

    def get_cost(self):
        """Calculate the total cost of the current solution to an instance"""
        self.cost = self._get_cost(self.routes)

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
        instance = {'cluster': cluster,
                    'dimension': len(cluster),
                    'distance': self.distance[np.ix_(cluster, cluster)],
                    'coords': self.coords[cluster]}
        return TSPInstance(instance)

    def polar_coord(self):
        """Calculate the polar co-ordinate, and sort with a reference to the node id."""
        depot = self.coords[0]
        polars = np.append(0, np.arctan((self.coords[1:, 1] - depot[1]) / (self.coords[1:, 0] - depot[0])))
        index = np.arange(polars.shape[0])  # create index array for indexing
        polars2 = np.c_[polars, index]
        self.polars = polars2[polars2[:, 0].argsort()]


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

    @staticmethod
    def _pos_check(route, node):
        """Check where in a route a node appears"""
        if route[0] == node:
            return 0
        elif route[-1] == node:
            return 1
        else:
            return 2
