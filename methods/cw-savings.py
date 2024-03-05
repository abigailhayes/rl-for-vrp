import vrplib
from operator import itemgetter
from utils import VRPInstance

data = vrplib.read_instance('instances/A/A-n32-k5.vrp')
solution = vrplib.read_solution('instances/A/A-n32-k5.sol')

class CWSavings(VRPInstance):
    """A class for implementing Clarke-Wright Savings on a VRP instance."""

    def route_init(self):
        """Initialise routes specific to CW Savings"""
        self.routes = [[i] for i in range(1,self.dimension)]

    def _calc_saving(self, i, j):
        """Calculate saving for a specific node pair"""
        return self.distance[i][0]+self.distance[0][j]-self.distance[i][j]

    def get_savings(self):
        self.savings = []
        for i in range(2, self.dimension):
            for j in range(1, i):
                self.savings.append((i, j, self._calc_saving(i, j)))
        self.savings.sort(key=itemgetter(2), reverse=True)

    def _get_route(self, i):
        """Get the current route the nodes of interest is in"""
        return [r for r in self.routes if i in r][0]

    def _pos_check(self, i):
        """Check where in a route a node appears"""
        route = self._get_route(i)
        if route[0]==i:
            return 0
        elif route[-1]==i:
            return 1
        else:
            return 2

    def _merge_routes(self, i, j):
        """Merge routes based on positioning of nodes"""
        if self._pos_check(i)<self._pos_check(j):
            return self._get_route(j)+self._get_route(i)
        elif self._pos_check(i)<self._pos_check(j):
            return self._get_route(i) + self._get_route(j)
        else:
            return self._get_route(i) + list(reversed(self._get_route(j)))

    def _cap_check(self, new_route):
        """Check that the new proposed route fits within the capacity demand"""
        return sum([self.demand[i] for i in new_route])

    def routing(self):
        for i, j, c in self.savings:
            if self._pos_check(i) == 2 or self._pos_check(j) == 2:
                continue
            if i in self._get_route(j):
                continue
            new_route = self._merge_routes(i, j)
            if self._cap_check(new_route) > self.capacity:
                continue
            self.routes = [r for r in self.routes if i not in r and j not in r]
            self.routes.append(new_route)

    def run_all(self):
        self.route_init()
        self.get_savings()
        self.routing()
        self.get_cost()
        if self.sol==True:
            self.compare_cost()

