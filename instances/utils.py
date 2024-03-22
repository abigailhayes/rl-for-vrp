import vrplib


def import_instance(folder, name):
    output = {'instance': vrplib.read_instance(f'{folder}/{name}.vrp'),
              'solution': vrplib.read_solution(f'{folder}/{name}.sol')}
    return output
