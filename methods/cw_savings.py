from operator import itemgetter
import methods.utils as utils

class CWSavings(utils.VRPInstance):
    """A class for implementing Clarke-Wright Savings on a VRP instance."""

    def route_init(self):
        """Initialise routes specific to CW Savings"""
        self.routes = [[i] for i in range(1,self.dimension)]

    def _calc_saving(self, i, j):
        """Calculate saving for a specific node pair"""
        return self.distance[i][0]+self.distance[0][j]-self.distance[i][j]

    def get_savings(self):
        """Calculated all savings"""
        self.savings = []
        for i in range(2, self.dimension):
            for j in range(1, i):
                self.savings.append((i, j, self._calc_saving(i, j)))
        self.savings.sort(key=itemgetter(2), reverse=True)

    def _merge_routes(self, node_pair):
        """Merge routes based on positioning of nodes"""
        if node_pair.pos_i<node_pair.pos_j:
            return node_pair.route_j+node_pair.route_i
        elif node_pair.pos_i<node_pair.pos_j:
            return node_pair.route_i + node_pair.route_j
        else:
            return node_pair.route_i + list(reversed(node_pair.route_j))

    def routing(self, talk=False):
        """Running the main part of the algorithm to provide a final route.
        Consider each node pair in turn, and join the routes if appropriate."""
        for i, j, c in self.savings:
            node_pair = utils.NodePair(i, j, self.routes)

            if node_pair.pos_i == 2 or node_pair.pos_j == 2:
                continue
            if i in node_pair.route_j:
                continue
            new_route = self._merge_routes(node_pair)
            if self._cap_check(new_route) > self.capacity:
                continue
            if talk==True:
                print("Current routes:", self.routes, "Join:", i, ", ", j, " Save: ", c)
            self.routes = [r for r in self.routes if i not in r and j not in r]
            self.routes.append(new_route)

    def run_all(self, talk=False):
        self.route_init()
        self.get_savings()
        self.routing(talk)
        self.get_cost()
        if self.sol==True:
            self.compare_cost()

