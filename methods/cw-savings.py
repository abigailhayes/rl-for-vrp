import vrplib
from operator import itemgetter
from utils import VRPInstance, NodePair

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

    def _merge_routes(self):
        """Merge routes based on positioning of nodes"""
        if self.node_pair.pos_i<self.node_pair.pos_j:
            return self.node_pair.route_j+self.node_pair.route_i
        elif self.node_pair.pos_i<self.node_pair.pos_j:
            return self.node_pair.route_i + self.node_pair.route_j
        else:
            return self.node_pair.route_i + list(reversed(self.node_pair.route_j))

    def _cap_check(self, new_route):
        """Check that the new proposed route fits within the capacity demand"""
        return sum([self.demand[i] for i in new_route])

    def routing(self):
        for i, j, c in self.savings:
            self.node_pair = NodePair(i, j, self.routes)

            if self.node_pair.pos_i == 2 or self.node_pair.pos_j == 2:
                continue
            if i in self.node_pair.route_j:
                continue
            new_route = self._merge_routes()
            if self._cap_check(new_route) > self.capacity:
                continue
            self.routes = [r for r in self.routes if i not in r and j not in r]
            self.routes.append(new_route)

    def run_all(self):
        self.route_init()
        self.get_savings()
        self.routing()
        del self.node_pair
        self.get_cost()
        if self.sol==True:
            self.compare_cost()

