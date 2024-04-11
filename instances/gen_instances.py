import random
import numpy as np
import vrplib


def gen_cvrp(filepath, nodes, capacity, max_demand, node_type='random', depot_type='centre'):
    # Initial values
    output = {'name': f'{node_type}-{depot_type}-{nodes}-{capacity}-{max_demand}',
              'type': 'CVRP',
              'dimension': nodes + 1,
              'edge_weight_type': 'EUC_2D',
              'capacity': capacity}

    # Helper function for generating coordinates
    def random_coords():
        return [random.random() for _ in range(2)]

    # Set depot location and demand
    if depot_type == 'centre':
        coords = [[0.5, 0.5]]
    elif depot_type == 'random':
        coords = [random_coords()]
    demand = [0]

    # Node co-ordinates and demand
    for i in range(nodes):
        coords.append(random_coords())
        demand.append(random.randint(1, max_demand))
    output['node_coord'] = np.array(coords)
    output['demand'] = np.array(demand)

    # Indicate depot position in the arrays
    output['depot'] = [0]

    # Edge weight (distances)
    distances = np.zeros((nodes + 1, nodes + 1))
    for i in range(nodes + 1):
        distances[i] = np.linalg.norm(output['node_coord'] - output['node_coord'][i], axis=1)
    output['edge_weight'] = distances

    vrplib.write_instance(f"{filepath}/{output['name']}", output)
