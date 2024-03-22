from methods.TSP.utils import TSPInstance
import random
import numpy as np


class GENI(TSPInstance):
    """Class to handle GENI procedure for TSP problems"""

    def __init__(self, instance, p=4):
        super().__init__(instance)
        self.full_route = False
        self.p = p
        self.route = random.sample(self.cluster, k=3)
        self.p_hoods = {}

    def _calc_p_hood(self, node):
        """Calculate the nodes in a p neighbourhood of the specified node."""
        if node not in self.route:
            raise ValueError(f'Node {node} not in current route.')

        if len(self.route) <= self.p:
            self.p_hoods[node] = [i for i in self.route if i != node]
        else:
            self.p_hoods[node] = [self.route[i] for i in np.argsort(self.distance[self.cluster.index(node)][[
                self.cluster.index(i) for i in self.route]])[1:self.p+1]]
