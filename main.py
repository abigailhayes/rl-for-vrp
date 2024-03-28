import instances.utils as instances_utils
import methods.cw_savings as cw_savings
from methods.or_tools import ORtools
from methods.sweep import Sweep
import utils

data = instances_utils.import_instance('instances/CVRP/A', 'A-n32-k5')

test = ORtools(data['instance'])
test.add_sol(data['solution'])
test.run_all()
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))

# Run over all test sets
utils.avg_perf('CVRP', 'CWSavings')

# Create TSP instance
import numpy as np
from methods.TSP.utils import TSPInstance
instance = {'cluster': [0] + test.clusters[0],
            'dimension': len([0] + test.clusters[0]),
            'distance': test.distance[np.ix_([0] + test.clusters[0], [0] + test.clusters[0])],
            'coords': test.coords[[0] + test.clusters[0]]}
tsp_test = TSPInstance(instance)