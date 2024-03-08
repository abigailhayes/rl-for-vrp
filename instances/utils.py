import vrplib

def import_instance(folder, name):
    output = {}
    output['instance'] = vrplib.read_instance(f'{folder}/{name}.vrp')
    output['solution'] = vrplib.read_solution(f'{folder}/{name}.sol')
    return output