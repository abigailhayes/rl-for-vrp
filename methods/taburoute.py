from operator import itemgetter
import methods.utils as utils


class CWSavings(utils.VRPInstance):
    """A class for implementing Clarke-Wright Savings on a VRP instance."""

    def __init__(self, instance):
        super().__init__(instance)
