from operator import itemgetter
import methods.utils as utils
from methods.TSP.utils import TSPInstance
import numpy as np

class Sweep(utils.VRPInstance):
    """A class for implementing Sweep on a VRP instance."""

    def polar_coord(self):
        """Calculate the polar co-ordinate, and sort with a reference to the node id."""
        depot = self.coords[0]
        polars = np.append(0,np.arctan((self.coords[1:,1]-depot[1])/(self.coords[1:,0]-depot[0])))
        index = np.arange(polars.shape[0])  # create index array for indexing
        polars2 = np.c_[polars, index]
        self.polars = polars2[polars2[:, 0].argsort()]

    def build_clusters(self):
        """Build the clusters via a sweep"""
        self.clusters = []
        new_route = []
        for coord, id in self.polars:
            if id==0:
                continue
            new_route.append(int(id))
            if self._cap_check(new_route) > self.capacity:
                # When capacity is exceeded, start new cluster
                self.clusters.append(new_route[:-1])
                new_route = [int(id)]
        self.clusters.append(new_route) # Save final cluster when no more nodes

    def routing(self, method):
        """Carry out routing for specified method"""
        self.routes = []
        for cluster in self.clusters:
            instance = self._gen_tsp_instance(cluster)
            getattr(instance, method)()
            self.routes.append(instance.route)

    def run_all(self, tsp_method):
        self.polar_coord()
        self.build_clusters()
        self.routing(tsp_method)
        self.get_cost()
        if self.sol==True:
            self.compare_cost()


