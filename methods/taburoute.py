import methods.utils as utils
from math import inf


class Taburoute(utils.VRPInstance):
    """A class for implementing Clarke-Wright Savings on a VRP instance."""

    def __init__(self, instance):
        super().__init__(instance)
        self.polar_coord()
        self.alpha = 1
        self.beta = 1
        self.f1_best = inf
