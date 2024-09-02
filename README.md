# Reinforcement Learning for the Vehicle Routing Problem

My project compares reinforcement learning methods for the vehicle routing problem (VRP) with more traditional methods. This is carried out for the standard capacitated VRP (CVRP) and for the variant with time windows (CVRP-TW).

## Instance generation

Code for generating instances is contained in [gen_instances](/instances/gen_instances.py). This has been used to produce test instances for the CVRP problem with a variety of features. The following can be varied:

- Depot location; central, edge or random
- Customer distribution; random or clustered
- Maximum demand; the maximum demand a customer can have
- Capacity; the capacity of the vehicles

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

Additionally, the sweep heuristic is implemented through [sweep.py](methods/sweep.py).

### Reinforcement learning

The first to be implemented is from the work of Nazari et al. and is based on [https://github.com/OptMLGroup/VRP-RL](https://github.com/OptMLGroup/VRP-RL). More information in [Nazari README](methods/nazari/README.md). There are issues with the validity of solutions returned by this method.

The remaining RL methods are implemented via the rl4co module in [rl4co_run.py](methods/rl4co_run.py). The methods implemented are 'am', 'amppo', 'pomo', 'symnco' and 'mdam'. Additionally, the customer nodes can first be clustered and then routed with travelling sales person (TSP) RL methods by specifying the method as 'rl4co_tsp'.

## Analysis

There is additional code to conduct analysis of the results.

* [best_or_tools.py](analysis/best_or_tools.py) searches through the OR tools results and identifies the optimal value for each instance
* [experiment_a.py](analysis/experiment_a.py) is for processing the results in comparison to the established CVRP benchmarks
* [experiment_b.py](analysis/experiment_b.py) is for processing the results in comparison to the generated CVRP instances
* [experiment_c.py](analysis/experiment_c.py) is for processing the results in comparison to the established CVRP-TW instances
* [instance_count.py](analysis/instance_count.py) looks at the number of solutions provided for each method
* [validity_check.py](analysis/validity_check.py) can be used to look at whether the RL methods have provided routes which visit every node exactly once without exceeding the capacity of any vehicles
* [am_custs.py](analysis/am_custs.py) has code for comparing results with different numbers of customers in training data
* [am_epochs.py](analysis/am_epochs.py) has code for comparing results with different numbers of epochs in training
* [plots_graphs.py](analysis/plots_graphs.py) has code for plotting various graphs of results
* [plots_instances.py](analysis/plots_instances.py) has code for drawing the routes produced by various methods
* [vehicles.py](analysis/vehicles.py) collects data on the number of vehicles used in the solutions provided by different methods

## Running code

Training and testing the models or baselines is done using [main.py](main.py). This can be run in the command line, with the experimental details set based on the information in parse_experiment() in [utils.py](utils.py).