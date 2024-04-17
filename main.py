# Import

from methods.or_tools import ORtools
import utils


def main():
    args = utils.parse_experiment()

    # Looking at current ids in use as folders
    id_list = [int(str.replace(item, 'exp_', '')) for item in os.listdir('results') if 'run' in item]

    # determine ID of this run
    if len(id_list) == 0:
        id = 1
    else:
        id = max(id_list) + 1