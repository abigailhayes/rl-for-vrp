# Copyright 2010-2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from the original to be included in a wider class

import methods.utils as utils

import random
from math import ceil
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


class ORtools(utils.VRPInstance):
    """A class for implementing the methods included in OR tools on a VRP instance."""

    def __init__(self, instance, init_method, improve_method=None):
        super().__init__(instance)
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.depot = 0
        self.init_method = init_method
        self.improve_method = improve_method
        if self.improve_method is not None:
            self.method = f'{self.improve_method}/{self.init_method}'
        else:
            self.method = self.init_method
        random.seed(self.seed)

    def print_solution(self, solution):
        """Prints solution on console."""
        print(f"Objective: {solution.ObjectiveValue()/1000}")
        total_distance = 0
        total_load = 0
        for vehicle_id in range(self.no_vehicles):
            index = self.routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            route_load = 0
            route = []
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route.append(node_index)
                route_load += self.demand[node_index]
                plan_output += f" {node_index} Load({route_load}) -> "
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )/1000
            plan_output += f" {self.manager.IndexToNode(index)} Load({route_load})\n"
            plan_output += f"Distance of the route: {route_distance}m\n"
            plan_output += f"Load of the route: {route_load}\n"
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
            route.remove(0)
            self.routes.append(route)
        print(f"Total distance of all routes: {total_distance}m")
        print(f"Total load of all routes: {total_load}")

    def _distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return int(self.distance[from_node][to_node]*1000)

    def _demand_callback(self, from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        return self.demand[from_node]

    def _min_vehicles(self):
        """Minimum number of vehicles that should be attempted"""
        self.no_vehicles = 2*ceil(sum(self.demand)/self.capacity)

    def _veh_capacities(self):
        """Generate a list of the correct length from a single capacity value"""
        return [self.capacity]*self.no_vehicles

    def setup(self):
        """Initiate all of the elements needed for running solutions"""
        # Managing indices conversion between node numbers and where the data is held
        self.manager = pywrapcp.RoutingIndexManager(self.dimension, self.no_vehicles, self.depot)
        self.routing = pywrapcp.RoutingModel(self.manager)
        # Define cost of each arc
        self.routing.SetArcCostEvaluatorOfAllVehicles(self.routing.RegisterTransitCallback(self._distance_callback))
        # Add in capacity limitations
        self.routing.AddDimensionWithVehicleCapacity(
            self.routing.RegisterUnaryTransitCallback(self._demand_callback),
            0,  # null capacity slack
            self._veh_capacities(),  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

    def search_settings(self):
        """Set up conditions for the search"""
        if self.init_method == 'savings':
            self.search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
            self.search_parameters.savings_parallel_routes = True
        elif self.init_method == 'cheapest_arc':
            self.search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        elif self.init_method == 'christofides':
            self.search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES
        elif self.init_method == 'local_cheapest_insert':
            self.search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION

        if self.improve_method == 'guided_local':
            self.search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        elif self.improve_method == 'sa':
            self.search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        elif self.improve_method == 'tabu':
            self.search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH

        self.search_parameters.log_search = False
        self.search_parameters.time_limit.FromSeconds(60)

    def run_all(self):
        self._min_vehicles()
        self.setup()
        self.search_settings()
        solution = self.routing.SolveWithParameters(self.search_parameters)
        self.print_solution(solution)
        self.get_cost()
        if self.sol:
            self.compare_cost()
