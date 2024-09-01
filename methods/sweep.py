import methods.utils as utils
import numpy as np


class Sweep(utils.VRPInstance):
    """A class for implementing Sweep on a VRP instance."""

    def __init__(self, instance):
        super().__init__(instance)
        self.polar_coord()
        self.build_clusters()
        for _ in range(10):
            self.build_rotate_clusters()

    def build_clusters(self):
        """Build the clusters via a sweep"""
        self.clusters = []
        new_route = []
        for coord, ident in self.polars:
            if ident == 0:
                continue
            new_route.append(int(ident))
            if self._cap_check(new_route) > self.capacity:
                # When capacity is exceeded, start new cluster
                self.clusters.append(new_route[:-1])
                new_route = [int(ident)]
        self.clusters.append(new_route)  # Save final cluster when no more nodes

    def build_rotate_clusters(self):
        """Try different rotations of the polar coordinates, and if clustering ia better, replace original"""
        # First change the starting angle for the sweep
        polars_new = self.polars.copy()
        polars_new[:, 0] += np.random.uniform(-np.pi, np.pi)
        polars_new[:, 0] = np.mod(polars_new[:, 0] + np.pi, 2 * np.pi) - np.pi
        polars_new.sort(axis=0)

        clusters = []
        new_route = []
        for coord, ident in polars_new:
            if ident == 0:
                continue
            new_route.append(int(ident))
            if self._cap_check(new_route) > self.capacity:
                # When capacity is exceeded, start new cluster
                clusters.append(new_route[:-1])
                new_route = [int(ident)]
        clusters.append(new_route)
        if len(clusters) < len(self.clusters):
            self.clusters = clusters

    def routing(self, tsp_type, method, improvement=None):
        """Carry out routing for specified method"""
        self.routes = []
        for cluster in self.clusters:
            instance = self._gen_tsp_instance(cluster, tsp_type)
            getattr(instance, method)()
            if improvement is not None:
                getattr(instance, improvement)()
            self.routes.append(instance.route)

    def run_all(self, tsp_type, tsp_method, tsp_improve=None):
        self.routing(tsp_type, tsp_method, tsp_improve)
        self.get_cost()
        if self.sol:
            self.compare_cost()
