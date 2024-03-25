from methods.TSP.utils import TSPInstance
import random
import numpy as np
from itertools import pairwise, combinations


class GENI(TSPInstance):
    """Class to handle GENI procedure for TSP problems"""

    def __init__(self, instance, p=5):
        super().__init__(instance)
        self.full_route = False
        self.p = p
        self.route = random.sample(self.cluster, k=3)
        self.p_hoods = {}

    def _calc_p_hood(self, node):
        """Calculate the nodes in a p neighbourhood of the specified node."""
        if node not in self.route:
            route = self.route + [node]
        else:
            route = self.route

        if len(self.route) <= self.p:
            return [i for i in self.route if i != node]
        else:
            output = [route[i] for i in np.argsort(self.distance[self.cluster.index(node)][[
                self.cluster.index(i) for i in route]])[:self.p+1]]
            output.remove(node)
            return output

    def _calc_p_hoods_route(self):
        """Calculate the p neighbourhoods for all nodes in the route"""
        for node in self.route:
            self.p_hoods[node] = self._calc_p_hood(node)

    @staticmethod
    def _next_item(route, item, up=True):
        """Find the next item in a list, by default going as read, but can be reversed"""
        if up:
            return route[(route.index(item) + 1) % len(route)]
        else:
            return route[(route.index(item) - 1) % len(route)]
    
    def _type1(self, i, j, node, best_insertion, reverse):
        """Attempts all possible Type I insertions"""
        if reverse:
            route = list(reversed(self.route))
        else:
            route = self.route
        i_1, j_1 = self._next_item(route, i, True), self._next_item(route, j, True)
        for k in self.p_hoods[i_1]:
            if k not in [i, j]:
                k_1 = self._next_item(route, k, True)
                test_route = route[:route.index(i_1)] + [node] + list(reversed(route[route.index(i_1):route.index(j_1)])) + list(reversed(route[route.index(j_1):route.index(k_1)])) + route[route.index(k_1):]
                if len(test_route) > len(self.route)+1:
                    continue
                cost = self._get_cost(test_route)
                if cost < best_insertion['cost']:
                    best_insertion['cost'] = cost
                    best_insertion['route'] = test_route
        return best_insertion

    def _type2(self, i, j, node, best_insertion, reverse):
        """Attempts all possible Type II insertions"""
        if reverse:
            route = list(reversed(self.route))
        else:
            route = self.route
        i_1, j_1 = self._next_item(route, i, True), self._next_item(route, j, True)
        for k in self.p_hoods[i_1]:
            if k not in [j, j_1]:
                for l in self.p_hoods[j_1]:
                    if l not in [i, i_1]:
                        # Check Type II insertions
                        test_route = route[:route.index(i_1)] + [node] + list(reversed(route[route.index(l):route.index(j_1)])) + route[route.index(j_1):route.index(k)] + list(reversed(route[route.index(i_1):route.index(l)])) + route[route.index(k):]
                        if len(test_route) > len(self.route) + 1:
                            continue
                        cost = self._get_cost(test_route)
                        if cost < best_insertion['cost']:
                            best_insertion['cost'] = cost
                            best_insertion['route'] = test_route

        return best_insertion

    def _add_node(self, node):
        """Carrying out one loop of Step 2 in the algorithm, adding a node"""
        p_hood = self._calc_p_hood(node)
        best_insertion = {'cost': np.sum(self.distance)}
        # Check all direct insertions to an edge with a node in neighbourhood
        for n, (i, j) in enumerate(pairwise(self.route + [self.route[0]])):
            if i in p_hood or j in p_hood:
                cost = self._get_cost(self.route[:n+1]+[node]+self.route[n+1:])
                if cost < best_insertion['cost']:
                    best_insertion['cost'] = cost
                    best_insertion['route'] = self.route[:n+1]+[node]+self.route[n+1:]
        # Now check all more complex insertions
        for i, j in combinations(p_hood, 2):
            if self._next_item(self.route, i, True) == j or self._next_item(self.route, i, False) == j:
                continue
            # Type I
            best_insertion = self._type1(i, j, node, best_insertion, False)
            best_insertion = self._type1(j, i, node, best_insertion, True)
            # Type II
            best_insertion = self._type2(i, j, node, best_insertion, False)
            best_insertion = self._type2(j, i, node, best_insertion, True)
        # Actually add in the node
        self.route = best_insertion['route']
        self._calc_p_hoods_route()

    def _standardise(self):
        self.route = self.route[self.route.index(0)+1:] + self.route[:self.route.index(0)]

    def run_all(self):
        """Running the whole algorithm"""
        self._calc_p_hoods_route()
        for node in self.cluster:
            if node in self.route:
                continue
            else:
                self._add_node(node)
        self._standardise()
