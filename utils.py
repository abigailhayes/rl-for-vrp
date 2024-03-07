import os

def get_dir(task):
    """Specifies the directory to run through, based on the task."""
    if task=='CVRP':
        return './instances/CVRP'
    else:
        raise ValueError("Unrecognised task")

def get_method(method):

def avg_perf(task, method):
    """Function to run over all available instances and get the average percentage that the algorithm
    is worse by
    Specify:
    - task; CVRP or other
    - method; the algorithm being tested"""
    directory = get_dir(task)
    for subdir in next(os.walk(directory))[1]:
        for example in [example for example in next(os.walk(f'{directory}/{subdir}'))[2] if example.endswith('vrp')]:
            print(example)
