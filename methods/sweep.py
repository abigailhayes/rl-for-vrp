from operator import itemgetter
import methods.utils as utils
import numpy as np

class Sweep(utils.VRPInstance):
    """A class for implementing Sweep on a VRP instance."""

    def polar_coord(self):
        """Calculates the polar co-ordinate, and sorts with a reference to the node id."""
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
                self.clusters.append(new_route[:-1])
                new_route = [int(id)]
        self.clusters.append(new_route)


