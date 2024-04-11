# Reinforcement Learning for the Vehicle Routing Problem

My project looks at comparing reinforcement learning methods for the vehicle routing problem (VRP) with more established methods. This is currently implemented for the standard capacitated VRP (CVRP).

## Instance generation

There will be code to make comparisons to established benchmarks, but also to specifically generated instance sets to test certain features. Code for generating instances is contained in [gen_instances](/instances/gen_instances.py).

## Solution methods

The main portion of established methods are implemented using Google OR tools in [or_tools](methods/or_tools.py).

