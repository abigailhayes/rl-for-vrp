
class TSPInstance:
    """A class for storing TSP instances
    - instance: need to provide an instance as input at creation"""
    def __init__(self, instance):
        self.distance = instance['edge_weight']
        self.dimension = instance['dimension']
        self.coords = instance['node_coord']
        self.route = []