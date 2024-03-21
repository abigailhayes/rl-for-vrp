import numpy as np
from itertools import pairwise


class TSPInstance:
    """A class for storing TSP instances
    - instance: need to provide an instance as input at creation"""
    def __init__(self, instance):
        self.cost = None
        self.cluster = instance['cluster']
        self.distance = instance['distance']
        self.dimension = instance['dimension']
        self.coords = instance['coords']
        self.route = []

    def _get_cost(self, route):
        """Calculate the total cost of a solution to an instance"""
        costs = 0
        pairs = list(pairwise([0]+route+[0]))
        for i, j in pairs:
            costs += self.distance[self.cluster.index(i)+1][self.cluster.index(j)+1]
        return costs

    def get_cost(self):
        """Calculate the total cost of the current solution to an instance"""
        self.cost = self._get_cost(self.route)

    def nearest_neighbour(self):
        """Constructs a route based on the nearest neighbour of the previous node"""
        self.route = []
        index = 0
        distance = self.distance.copy().view(np.ma.MaskedArray)
        distance[:, index] = np.ma.masked
        # Continue to find the nearest node to the previous node, until all are included
        while len(self.route) < self.dimension-1:
            index = distance[index].argmin()
            self.route.append(self.cluster[index])
            distance[:, index] = np.ma.masked

    def _insertion(self, method):
        """Constructs a route based on the nearest of furthest insertion to the existing tour"""
        distance = self.distance.copy().view(np.ma.MaskedArray)
        route, indices = [0, 0], [0, 0]

        index = 0                          # Track latest node under consideration
        distance[:, index] = np.ma.masked  # Mask the column for node that has been added to route

        # Add first non-depot node, also store index, and mask in distances
        if method == 'nearest':
            index = distance[route].argmin()
        elif method == 'furthest':
            index = distance[route].argmax()
        else:
            raise ValueError('Unsupported method.')
        route.insert(1, self.cluster[index])
        indices.insert(1, index)
        distance[:, index] = np.ma.masked

        # Loop to add in all other nodes
        while len(route) < self.dimension+1:
            if method == 'nearest':
                index = np.unravel_index(distance[indices].argmin(), distance.shape)[1]
            elif method == 'furthest':
                index = np.unravel_index(distance[indices].argmax(), distance.shape)[1]
            # Calculate the cost for adding between each pair of nodes in the current route, and add in the optimal
            # place
            test_list = [
                self.distance[indices[i], index] + self.distance[index, indices[i + 1]] -
                self.distance[indices[i], indices[i + 1]] for i in range(len(route) - 1)]
            route.insert(np.argmin(test_list)+1, self.cluster[index])
            indices.insert(np.argmin(test_list)+1, index)
            distance[:, index] = np.ma.masked

        route.remove(0)
        route.remove(0)
        self.route = route

    def nearest_insertion(self):
        self._insertion('nearest')

    def furthest_insertion(self):
        self._insertion('furthest')

    def imp_2opt(self):
        self.get_cost()
        pairs = list(pairwise([0] + self.route + [0]))
