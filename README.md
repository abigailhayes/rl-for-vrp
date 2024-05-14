# Reinforcement Learning for the Vehicle Routing Problem

My project looks at comparing reinforcement learning methods for the vehicle routing problem (VRP) with more established methods. This is currently implemented for the standard capacitated VRP (CVRP).

## Instance generation

There will be code to make comparisons to established benchmarks, but also to specifically generated instance sets to test certain features. Code for generating instances is contained in [gen_instances](/instances/gen_instances.py).

## Solution methods

### General

The main portion of established methods are implemented using Google OR tools in [or_tools](methods/or_tools.py). 

This is set up to use the initial methods:
- savings - SAVINGS
- cheapest_arc - PATH_CHEAPEST_ARC
- christofides - CHRISTOFIDES
- local_cheapest_insert - LOCAL_CHEAPEST_INSERTION

And for the improvement methods:
- guided_local - GUIDED_LOCAL_SEARCH
- sa - SIMULATED_ANNEALING
- tabu - GENERIC_TABU_SEARCH

### Reinforcement learning

The first to be implemented is from the work of Nazari et al. and is based on [https://github.com/OptMLGroup/VRP-RL](https://github.com/OptMLGroup/VRP-RL). More information in [Nazari README](methods/nazari/README.md).
