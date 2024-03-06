import vrplib

def import_instance(folder, name):
    output = {}
    output['instance'] = vrplib.read_instance(f'instances/{folder}/{name}.vrp')
    output['solution'] = vrplib.read_solution(f'instances/{folder}/{name}.sol')
    return output