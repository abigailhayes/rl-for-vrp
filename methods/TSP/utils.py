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
        curr_ind = 0
        rem_nodes = self.cluster.copy()
        rem_nodes.remove(rem_nodes[curr_ind])
        distance = np.delete(self.distance.copy(), rem_nodes[curr_ind], axis=1)
        # Continue to find the nearest node to the previous node, whilst nodes remain
        while len(self.route)<self.dimension-1:
            curr_ind = distance[curr_ind].argmin()
            self.route.append(rem_nodes[curr_ind])
            rem_nodes.remove(rem_nodes[curr_ind])
            distance = np.delete(distance, curr_ind, axis=1)

    def nearest_insertion(self):
        """Constructs a route based on the nearest insertion to the existing tour"""
        route = [0]



