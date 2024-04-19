from collections import namedtuple

TaskVRP = namedtuple('TaskVRP', ['task_name',
                                 'input_dim',
                                 'n_nodes',
                                 'n_cust',
                                 'decode_len',
                                 'capacity',
                                 'demand_max'])

task_lst = {}

# Setting up my versions
small_high = TaskVRP(task_name='vrp',
                     input_dim=3,
                     n_nodes=11,
                     n_cust=10,
                     decode_len=16,
                     capacity=100,
                     demand_max=90)
task_lst['small_high'] = small_high

small_medium = TaskVRP(task_name='vrp',
                       input_dim=3,
                       n_nodes=11,
                       n_cust=10,
                       decode_len=16,
                       capacity=100,
                       demand_max=50)
task_lst['small_medium'] = small_medium

small_low = TaskVRP(task_name='vrp',
                    input_dim=3,
                    n_nodes=11,
                    n_cust=10,
                    decode_len=16,
                    capacity=100,
                    demand_max=30)
task_lst['small_low'] = small_low

med_high = TaskVRP(task_name='vrp',
                   input_dim=3,
                   n_nodes=21,
                   n_cust=20,
                   decode_len=16,
                   capacity=100,
                   demand_max=90)
task_lst['med_high'] = med_high

med_medium = TaskVRP(task_name='vrp',
                     input_dim=3,
                     n_nodes=21,
                     n_cust=20,
                     decode_len=16,
                     capacity=100,
                     demand_max=50)
task_lst['med_medium'] = med_medium

med_low = TaskVRP(task_name='vrp',
                  input_dim=3,
                  n_nodes=21,
                  n_cust=20,
                  decode_len=16,
                  capacity=100,
                  demand_max=30)
task_lst['med_low'] = med_low

high_high = TaskVRP(task_name='vrp',
                    input_dim=3,
                    n_nodes=51,
                    n_cust=50,
                    decode_len=16,
                    capacity=100,
                    demand_max=90)
task_lst['high_high'] = high_high

high_medium = TaskVRP(task_name='vrp',
                      input_dim=3,
                      n_nodes=51,
                      n_cust=50,
                      decode_len=16,
                      capacity=100,
                      demand_max=50)
task_lst['high_medium'] = high_medium

high_low = TaskVRP(task_name='vrp',
                   input_dim=3,
                   n_nodes=51,
                   n_cust=50,
                   decode_len=16,
                   capacity=100,
                   demand_max=30)
task_lst['high_low'] = high_low

giant_high = TaskVRP(task_name='vrp',
                     input_dim=3,
                     n_nodes=101,
                     n_cust=100,
                     decode_len=16,
                     capacity=100,
                     demand_max=90)
task_lst['giant_high'] = giant_high

giant_medium = TaskVRP(task_name='vrp',
                       input_dim=3,
                       n_nodes=101,
                       n_cust=100,
                       decode_len=16,
                       capacity=100,
                       demand_max=50)
task_lst['giant_medium'] = giant_medium

giant_low = TaskVRP(task_name='vrp',
                    input_dim=3,
                    n_nodes=101,
                    n_cust=100,
                    decode_len=16,
                    capacity=100,
                    demand_max=30)
task_lst['giant_low'] = giant_low

# VRP10
vrp10 = TaskVRP(task_name='vrp',
                input_dim=3,
                n_nodes=11,
                n_cust=10,
                decode_len=16,
                capacity=20,
                demand_max=9)
task_lst['vrp10'] = vrp10

# VRP20
vrp20 = TaskVRP(task_name='vrp',
                input_dim=3,
                n_nodes=21,
                n_cust=20,
                decode_len=30,
                capacity=30,
                demand_max=9)
task_lst['vrp20'] = vrp20

# VRP50
vrp50 = TaskVRP(task_name='vrp',
                input_dim=3,
                n_nodes=51,
                n_cust=50,
                decode_len=70,
                capacity=40,
                demand_max=9)
task_lst['vrp50'] = vrp50

# VRP100
vrp100 = TaskVRP(task_name='vrp',
                 input_dim=3,
                 n_nodes=101,
                 n_cust=100,
                 decode_len=140,
                 capacity=50,
                 demand_max=9)
task_lst['vrp100'] = vrp100
