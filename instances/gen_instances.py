import random
import numpy as np

def gen_cvrp(type, nodes, capacity, max_demand):
    # Initial values
    output = {'name': f'{type}-{nodes}-{capacity}-{max_demand}',
              'type': 'CVRP',
              'dimension': nodes+1,
              'edge_weight_type': 'EUC_2D',
              'capacity': capacity}

    # Helper function for generating coordinates
    def random_coords():
        return [random.random() for i in range(1)]

    # Set depot location and demand
    if type == 'random_centre':
        coords = [[0.5,0.5]]
    elif type == 'random_random':
        coords = [random_coords()]
    demand = [0]

    # Node co-ordinates and demand
    for i in range(nodes):
        coords.append(random_coords())
        demand.append(random.randint(1,max_demand))
    output['node_coord'] = np.array(coords)
    output['demand'] = np.array(demand)

    # Indicate depot position in the arrays
    output['depot'] = [0]

    # Edge weight (distances)


    return output

# name type-nodes-capacity-max_demand
# type CVRP
# dimension
# edge_weight_type
# capacity
# node_coord
# demand
# depot [0]
# edge_weight