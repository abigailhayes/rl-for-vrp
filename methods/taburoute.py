import random

import methods.utils as utils
import numpy as np

from methods.TSP.genius import GENI


class Taburoute(utils.VRPInstance):
    """A class for implementing Taburoute on a VRP instance."""

    def __init__(self, instance, p=4):
        super().__init__(instance)
        self.polar_coord()
        self.alpha = 1
        self.p = p
        self._init_sol()

    def _f2(self):
        return self.cost + self.alpha*sum([max(0, self._cap_check(route)-self.capacity) for route in self.routes])

    def _init_sol(self):
        """Create initial solution by following GENI for all nodes, and then splitting to meet capacity constraints"""
        # First use GENI to give an initial solution
        self.routes = []
        i = random.randint(1, self.dimension-1)
        cluster = [0] + list(range(i, self.dimension)) + list(range(1, i))
        instance = {'cluster': cluster,
                    'dimension': self.dimension,
                    'distance': self.distance[np.ix_(cluster, cluster)],
                    'coords': self.coords[cluster]}
        tsp_instance = GENI(instance, self.p)
        tsp_instance.run_all()
        single_route = tsp_instance.route
        # Now split the solution into capacity respecting routes
        new_route = []
        for item in single_route:
            new_route.append(item)
            if self._cap_check(new_route) > self.capacity:
                # When capacity is exceeded, start new route
                self.routes.append(new_route[:-1])
                new_route = [item]
        self.routes.append(new_route)  # Save final route when no more nodes
        # Saving all info to initialise variables
        self.get_cost()
        self.f1_best = self.cost       # Best solution so far to objective fn
        self.f2_best = self._f2()      # Best solution so far with penalty for infeasibility
        if self._cap_check(self.routes[-1]) <= self.capacity:
            self.s_best = self.routes      # Best feasible route so far
        else:
            self.s_best = None
        self.s_best_all = self.routes  # Best route so far, feasible or infeasible
