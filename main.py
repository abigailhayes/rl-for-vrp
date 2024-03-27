import instances.utils as instances_utils
import methods.cw_savings as cw_savings
from methods.sweep import Sweep
from methods.taburoute import Taburoute
import utils

data = instances_utils.import_instance('instances/CVRP/A', 'A-n80-k10')

test = Taburoute(data['instance'])
test.add_sol(data['solution'])


test.run_all('standard', 'nearest_insertion')
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))
test.run_all('standard', 'furthest_insertion')
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))
test.run_all('standard', 'nearest_neighbour')
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))
test.run_all('GENI', 'geni')
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))
test.run_all('GENI', 'genius')
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))

# Run over all test sets
#utils.avg_perf('CVRP', 'CWSavings')

# Create TSP instance
import numpy as np
from methods.TSP.utils import TSPInstance
from methods.TSP.genius import GENI
instance = {'cluster': [0] + test.clusters[7],
            'dimension': len([0] + test.clusters[7]),
            'distance': test.distance[np.ix_([0] + test.clusters[7], [0] + test.clusters[7])],
            'coords': test.coords[[0] + test.clusters[7]]}
tsp_test = GENI(instance)
tsp_test.run_all()