import random
import numpy as np
import vrplib
import os
from itertools import product


def gen_cvrp(filepath, ident, nodes, capacity, max_demand, node_type='random', depot_type='centre'):
    """Function for generating a single CVRP instance - filepath - the folder where the instance should be saved -
    ident - sets an identifier - nodes - number of customer nodes - capacity - the capacity of each of the vehicles -
    max_demand - the maximum demand of any one customer - node_type - how the co-ordinates of the customers should be
    selected, either completely randomly or with clustering - depot_type - the position of the depot, can be random
    or centred"""
    # Initial values
    output = {'NAME': f'{node_type}-{depot_type}-{nodes}-{capacity}-{max_demand}-{ident}',
              'TYPE': 'CVRP',
              'DIMENSION': nodes + 1,
              'EDGE_WEIGHT_TYPE': 'EUC_2D',
              'CAPACITY': capacity}

    # Helper function for generating coordinates
    def random_coords():
        return [random.random() for _ in range(2)]

    # Set depot location and demand
    coords = [random_coords()]
    if depot_type == 'centre':
        coords = [[0.5, 0.5]]
    elif depot_type == 'outer':
        element = random.randint(0, 1)
        coords[0][element] = round(coords[0][element])
    demand = [0]

    # Node co-ordinates and demand
    if node_type == 'random':
        for i in range(nodes):
            coords.append(random_coords())
            demand.append(random.randint(1, max_demand))
    elif node_type == 'clusters':
        no_clusters = random.randint(2, max(round(nodes / 10),2))
        centres = [random_coords() for _ in range(no_clusters)]
        for i in range(nodes):
            cluster = random.randint(0, no_clusters - 1)
            coords.append(np.clip([centres[cluster][i] + (random.random() - 0.5) / 3 for i in range(2)], 0, 1))
            demand.append(random.randint(1, max_demand))
    output['NODE_COORD_SECTION'] = np.array(coords)
    output['DEMAND_SECTION'] = np.array(demand)

    # Indicate depot position in the arrays
    output['DEPOT_SECTION'] = [1, -1]

    # Edge weight (distances)
    """distances = np.zeros((nodes + 1, nodes + 1))
    for i in range(nodes + 1):
        distances[i] = np.linalg.norm(output['NODE_COORD_SECTION'] - output['NODE_COORD_SECTION'][i], axis=1)
    output['EDGE_WEIGHT'] = distances"""

    # output['COMMENT'] = f'{no_clusters} clusters'

    vrplib.write_instance(f"{filepath}/{output['NAME']}.vrp", output)


def gen_cvrp_multi(set_type, seed, number=100, nodes=20, capacity=100, max_demand=20):
    """Setting up to generate instances with specific characteristics
    - set_type - the name used to refer overall to the type
    - seed - generation will be reproducible
    - number - how many instances to generate
    - nodes - number of customer nodes in each instance
    - capacity - defaults to being 100, but can be set to other values
    - max_demand - specifies the maximum demand"""
    # Set up folder
    filepath = f'instances/CVRP/generate/{set_type}-{nodes}-{max_demand}-{capacity}-{seed}'
    os.makedirs(filepath, exist_ok=True)
    random.seed(seed)

    # Generate instances
    if set_type == 'random_random':
        for i in range(number):
            gen_cvrp(filepath, i, nodes, capacity, max_demand, node_type='random', depot_type='random')
    elif set_type == 'random_centre':
        for i in range(number):
            gen_cvrp(filepath, i, nodes, capacity, max_demand, node_type='random', depot_type='centre')
    elif set_type == 'random_outer':
        for i in range(number):
            gen_cvrp(filepath, i, nodes, capacity, max_demand, node_type='random', depot_type='outer')
    elif set_type == 'cluster_random':
        for i in range(number):
            gen_cvrp(filepath, i, nodes, capacity, max_demand, node_type='clusters', depot_type='random')
    elif set_type == 'cluster_centre':
        for i in range(number):
            gen_cvrp(filepath, i, nodes, capacity, max_demand, node_type='clusters', depot_type='centre')
    elif set_type == 'cluster_outer':
        for i in range(number):
            gen_cvrp(filepath, i, nodes, capacity, max_demand, node_type='clusters', depot_type='outer')


def main():
    variants = ['random_random', 'random_centre', 'random_outer', 'cluster_random', 'cluster_centre', 'cluster_outer']
    custs = [10, 20, 50, 100]
    capacity = [100]
    max_demand = [90, 50, 30]
    for (variant, cust, cap, demand) in product(*[variants, custs, capacity, max_demand]):
        print(variant, cust, cap, demand)
        gen_cvrp_multi(variant, seed=42, number=100, nodes=cust, capacity=cap, max_demand=demand)


if __name__ == '__main__':
    main()
