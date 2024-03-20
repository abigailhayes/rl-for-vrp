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
        route = [0]



