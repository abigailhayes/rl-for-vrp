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

    def _calc_p_hood(self, route, node):
        """Calculate the nodes in a p neighbourhood of the specified node."""
        if node not in route:
            route = route + [node]

        if len(route) <= self.p:
            return [i for i in route if i != node]
        else:
            output = [route[i] for i in np.argsort(self.distance[self.cluster.index(node)][[
                self.cluster.index(i) for i in route]])[:self.p + 1]]
            output.remove(node)
            return output

    def _calc_p_hoods(self, route):
        """Calculate the p neighbourhoods for all nodes in the specified route"""
        p_hoods = {}
        for node in route:
            p_hoods[node] = self._calc_p_hood(route, node)
        return p_hoods

    def _calc_p_hoods_route(self):
        """Calculate the p neighbourhoods for all nodes in the route"""
        for node in self.route:
            self.p_hoods[node] = self._calc_p_hood(self.route, node)

    @staticmethod
    def _next_item(route, item, up=True):
        """Find the next item in a list, by default going as read, but can be reversed"""
        return route[(route.index(item) + 2 * up - 1) % len(route)]

    @staticmethod
    def _type1_string(route, i_1, j_1, k_1, node):
        """Type 1 stringing, need i+1, j+1 and k+1 nodes, and the node to be inserted"""
        return route[:route.index(i_1)] + [node] + list(reversed(
                route[route.index(i_1):route.index(j_1)])) + list(reversed(
                    route[route.index(j_1):route.index(k_1)])) + route[route.index(k_1):]

    @staticmethod
    def _type1_unstring(route, i_1, j_1, node):
        """Type 1 stringing, need i+1 and j+1, and the node to be inserted"""
        route = route[route.index(node):] + route[:route.index(node)]
        return (list(reversed(route[route.index(i_1):route.index(j_1)])
                     ) + route[route.index(j_1):] + list(reversed(route[1:route.index(i_1)])))

    def _type1_routes(self, route, i, j, node):
        """Generates all appropriate Type I insertion routes for a pair of nodes and a route of specific orientation"""
        i_1, j_1 = self._next_item(route, i, True), self._next_item(route, j, True)
        output = []
        p_hoods = self._calc_p_hoods(route)
        for k in p_hoods[i_1]:
            if route.index(k) < route.index(i) or route.index(k) > route.index(j):
                k_1 = self._next_item(route, k, True)
                test_route = self._type1_string(route, i_1, j_1, k_1, node)
                if len(test_route) > len(route) + 1:
                    continue
                else:
                    output.append(test_route)
        return output

    def _type1(self, i, j, node, best_insertion, route):
        """Attempts all possible Type I insertions for the specified nodes and route orientation"""
        test_routes = self._type1_routes(route, i, j, node)
        for test_route in test_routes:
            cost = self._get_cost(test_route)
            if cost < best_insertion['cost']:
                best_insertion['cost'] = cost
                best_insertion['route'] = test_route
        return best_insertion

    @staticmethod
    def _type2_string(route, i_1, j_1, k, m, node):
        """Type II stringing, need i+1, j+1, k and m nodes, and the node to be inserted"""
        return route[:route.index(i_1)] + [node] + list(reversed(
                route[route.index(m):route.index(j_1)])) + route[route.index(j_1):route.index(k)] + list(reversed(
                    route[route.index(i_1):route.index(m)])) + route[route.index(k):]

    @staticmethod
    def _type2_unstring(route, i, j_1, m_1, node):
        """Type II stringing, need i, j+1 and l+1 nodes, and the node to be inserted"""
        route = route[route.index(node):] + route[:route.index(node)]

        return route[route.index(i):route.index(j_1)] + route[route.index(m_1):] + list(reversed(
                route[route.index(j_1):route.index(m_1)])) + list(reversed(route[1:route.index(i)]))

    def _type2_routes(self, route, i, j, node):
        """Generates all appropriate Type II insertion routes for a pair of nodes and a route of specific orientation"""
        i_1, j_1 = self._next_item(route, i, True), self._next_item(route, j, True)
        output = []
        p_hoods = self._calc_p_hoods(route)
        for k in p_hoods[i_1]:
            if route.index(k) < route.index(i) or route.index(k) > route.index(j_1):
                for m in self.p_hoods[j_1]:
                    if route.index(i_1) < route.index(m) < route.index(j):
                        test_route = self._type2_string(route, i_1, j_1, k, m, node)
                        if len(test_route) > len(route) + 1:
                            continue
                        else:
                            output.append(test_route)
        return output

    def _type2(self, i, j, node, best_insertion, route):
        """Attempts all possible Type II insertions"""
        test_routes = self._type1_routes(route, i, j, node)
        for test_route in test_routes:
            cost = self._get_cost(test_route)
            if cost < best_insertion['cost']:
                best_insertion['cost'] = cost
                best_insertion['route'] = test_route
        return best_insertion

    def _add_node(self, route, node):
        """Carrying out one loop of Step 2 in the algorithm, adding a node"""
        p_hood = self._calc_p_hood(route, node)
        best_insertion = {'cost': np.sum(self.distance)}
        # Check all direct insertions to an edge with a node in neighbourhood
        for n, (i, j) in enumerate(pairwise(route + [route[0]])):
            if i in p_hood or j in p_hood:
                cost = self._get_cost(route[:n + 1] + [node] + route[n + 1:])
                if cost < best_insertion['cost']:
                    best_insertion['cost'] = cost
                    best_insertion['route'] = route[:n + 1] + [node] + route[n + 1:]
        # Now check all more complex insertions
        for i, j in combinations(p_hood, 2):
            if self._next_item(route, i, True) == j or self._next_item(route, i, False) == j:
                continue
            # Type I
            best_insertion = self._type1(i, j, node, best_insertion, route)
            best_insertion = self._type1(j, i, node, best_insertion, list(reversed(route)))
            # Type II
            best_insertion = self._type2(i, j, node, best_insertion, route)
            best_insertion = self._type2(j, i, node, best_insertion, list(reversed(route)))
        return best_insertion['route']

    def _us_single(self, route, node):
        """Apply unstringing and restringing for a single setting"""
        best_route = {'route': route, 'cost': self._get_cost(route)}
        p_hoods = self._calc_p_hoods(route)
        routes = [route, list(reversed(route))]
        for route in routes:
            for i, j in zip(p_hoods[self._next_item(route, node, False)], p_hoods[self._next_item(route, node, True)]):
                # Type I removals
                i_1, j_1 = self._next_item(route, i, True), self._next_item(route, j, True)
                if route.index(i) > route.index(j):
                    j_i_span = route[route.index(j):route.index(i)]
                else:
                    j_i_span = route[route.index(j):] + route[:route.index(i)]
                if len({i, j, i_1, j_1, node}) != 5 or node not in j_i_span:
                    continue
                else:
                    short_route = self._type1_unstring(route, i_1, j_1, node)
                    test_route = self._add_node(short_route, node)
                    if self._get_cost(test_route) < best_route['cost']:
                        best_route = {'route': test_route, 'cost': self._get_cost(test_route)}
                # Type II removals
                if route.index(i) > route.index(j):
                    i_j_span = route[route.index(i):]+route[:route.index(j)]
                else:
                    i_j_span = route[route.index(i):route.index(j)]
                for m in p_hoods[i_1]:
                    if route.index(m) in i_j_span:
                        m_1 = self._next_item(route, m, True)
                        short_route = self._type2_unstring(route, i, j_1, m_1, node)
                        test_route = self._add_node(short_route, node)
                        if self._get_cost(test_route) < best_route['cost']:
                            best_route = {'route': test_route, 'cost': self._get_cost(test_route)}
        return best_route['route'], best_route['cost']

    def us_improve(self):
        """US improvement method with unstringing and restringing"""
        route, route_best = self.route, self.route
        cost, cost_best = self._get_cost(route), self._get_cost(route)
        t = 0
        while t < len(self.route):
            route, cost = self._us_single(route, route[t])
            if cost < cost_best:
                t = 0
                cost_best = cost
                route_best = route
            else:
                t += 1
        self.route = route_best

    def _standardise(self):
        self.route = self.route[self.route.index(0) + 1:] + self.route[:self.route.index(0)]

    def geni(self):
        """Running the whole algorithm"""
        self._calc_p_hoods_route()
        for node in self.cluster:
            if node in self.route:
                continue
            else:
                self.route = self._add_node(self.route, node)
                self._calc_p_hoods_route()
        self._standardise()

    def genius(self):
        """Running the whole algorithm with improvement"""
        self._calc_p_hoods_route()
        for node in self.cluster:
            if node in self.route:
                continue
            else:
                self.route = self._add_node(self.route, node)
                self._calc_p_hoods_route()
        self.us_improve()
        self._standardise()
