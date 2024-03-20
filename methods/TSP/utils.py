import numpy as np

class TSPInstance:
    """A class for storing TSP instances
    - instance: need to provide an instance as input at creation"""
    def __init__(self, instance):
        self.cluster = instance['cluster']
        self.distance = instance['distance']
        self.dimension = instance['dimension']
        self.coords = instance['coords']
        self.route = []

    def nearest_neighbour(self):
        """Constructs a route based on the nearest neighbour of the previous node"""
        self.route = []
        index = 0
        distance = self.distance.copy().view(np.ma.MaskedArray)
        distance[:,index] = np.ma.masked
        # Continue to find the nearest node to the previous node, until all are included
        while len(self.route)<self.dimension-1:
            index = distance[index].argmin()
            self.route.append(self.cluster[index])
            distance[:,index] = np.ma.masked

    def nearest_insertion(self):
        """Constructs a route based on the nearest insertion to the existing tour"""
        distance = self.distance.copy().view(np.ma.MaskedArray)
        route, indices = [0, 0], [0, 0]

        index = 0                          # Track latest node under consideration
        distance[:, index] = np.ma.masked  # Mask the column for node that has been added to route

        # Add first non-depot node, also store index, and mask in distances
        index = distance[route].argmin()
        route.insert(1, self.cluster[index])
        indices.insert(1, index)
        distance[:, index] = np.ma.masked

        # Loop to add in all other nodes
        while len(route) < self.dimension+1:
            index = np.unravel_index(distance[indices].argmin(), distance.shape)[1]
            # Calculate the cost for adding between each pair of nodes in the current route, and add in the optimal place
            test_list = [
                self.distance[indices[i], index] + self.distance[index, indices[i + 1]] -
                self.distance[indices[i], indices[i + 1]] for i in range(len(route) - 2)]
            route.insert(np.argmin(test_list)+1, self.cluster[index])
            indices.insert(np.argmin(test_list)+1, index)
            distance[:, index] = np.ma.masked

        route.remove(0)
        route.remove(0)
        self.route = route





