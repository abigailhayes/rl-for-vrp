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
        self.route = []
        curr_node = self.cluster[0]
        rem_nodes = self.cluster.copy()
        rem_nodes.remove(curr_node)
        distance = np.delete(self.distance.copy(), curr_node, axis=1)
        while len(self.route)<self.dimension-1:
            curr_node = distance[curr_node].argmin()
            self.route.append(rem_nodes[curr_node])
            rem_nodes.remove(rem_nodes[curr_node])
            distance = np.delete(distance, curr_node, axis=1)


