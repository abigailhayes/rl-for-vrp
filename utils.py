import os
import instances.utils as instances_utils

def get_dir(task):
    """Specifies the directory to run through, based on the task."""
    if task=='CVRP':
        return './instances/CVRP'
    else:
        raise ValueError("Unrecognised task")

def apply_method(method, instance):
    """Apply the appropriate method to the example dataset."""
    if method=='CWSavings':
        output = cw_savings.CWSavings(instance['instance'])
    else:
        raise ValueError("Unrecognised method")

    output.add_sol(instance['solution'])
    output.run_all()
    return output

def avg_perf(task, method):
    """Function to run over all available instances and get the average percentage that the algorithm
    is worse by
    Specify:
    - task; CVRP or other
    - method; the algorithm being tested"""
    directory = get_dir(task)
    for subdir in next(os.walk(directory))[1]:
        for example in [example[:-4] for example in next(os.walk(f'{directory}/{subdir}'))[2] if example.endswith('vrp')]:
            instance = instances_utils.import_instance(subdir, example)
            run = apply_method(method, instance)
