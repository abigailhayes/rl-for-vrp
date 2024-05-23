import vrplib


def import_instance(folder, name):
    """Import standard CVRP instance, in VRPLIB format"""
    output = {'instance': vrplib.read_instance(f'{folder}/{name}.vrp'),
              'solution': vrplib.read_solution(f'{folder}/{name}.sol')}
    return output


def import_twinstance(folder, name):
    """Import CVRP-TW instance, in Solomon format"""
    output = {'instance': vrplib.read_instance(f'{folder}/{name}.txt', instance_format="solomon"),
              'solution': vrplib.read_solution(f'{folder}/{name}.sol')}
    return output


def shrink_twinstance(instance, size):
    """Deriving the smaller Solomon instances"""
    size = int(size)
    return {'name': instance['name']+'.'+str(size),
            'type': 'CVRPTW',
            'vehicles': instance['vehicles'],
            'capacity': instance['capacity'],
            'node_coord': instance['node_coord'][:size+1,],
            'demand': instance['demand'][:size+1,],
            'time_window': instance['time_window'][:size+1,],
            'service_time': instance['service_time'][:size+1,],
            'edge_weight': instance['edge_weight'][:size+1,:size+1],
            'dimension': size+1}
