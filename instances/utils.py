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
