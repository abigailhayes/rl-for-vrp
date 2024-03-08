from operator import itemgetter
import methods.utils as utils
import numpy as np

class Sweep(utils.VRPInstance):
    """A class for implementing Sweep on a VRP instance."""

    def polar_coord(self):
        depot = self.coords[0]
        self.polars = np.append(0,np.arctan((self.coords[1:,1]-depot[1])/(self.coords[1:,0]-depot[0])))