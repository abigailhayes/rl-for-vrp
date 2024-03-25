import methods.utils as utils
from math import inf
import numpy as np

from methods.TSP.genius import GENI


class Taburoute(utils.VRPInstance):
    """A class for implementing Clarke-Wright Savings on a VRP instance."""

    def __init__(self, instance):
        super().__init__(instance)
        self.polar_coord()
        self.alpha = 1
        self.beta = 1
        self.f1_best = inf

    def _init_sol(self):
        """Create initial solution by following GENI for all nodes, and then splitting to meet capacity constraints"""
        self.routes = []
        instance = {'cluster': list(range(self.dimension)),
                    'dimension': self.dimension,
                    'distance': self.distance,
                    'coords': self.coords}
        tsp_instance = GENI(instance)
        tsp_instance.run_all()
        single_route = tsp_instance.route

        new_route = []
        for item in single_route:
            new_route.append(item)
            if self._cap_check(new_route) > self.capacity:
                # When capacity is exceeded, start new route
                self.routes.append(new_route[:-1])
                new_route = [item]
        self.routes.append(new_route)  # Save final route when no more nodes
