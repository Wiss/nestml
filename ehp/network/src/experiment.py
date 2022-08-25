"""
experiment.py
    perform one experiment using the biophysical network and the yaml config
    file in config
"""
import datetime
import time
import yaml
import os
import copy

import src.network
#from utils.figures import create_weights_figs, \
#create_frs_figs, create_graph_figs
#
from utils.manage_files import create_folder, load_config, save_config, \
    save_data, load_data
#from utils.bionet_tools import weight_init, population_pos_init

#from logging.logger import logger TODO

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="microcircuit experiment",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", type=str, required=True,
            help="Configuration file. \
            See config\ folder")
    args = parser.parse_args()
    config_path = args.file

    ## CONFIG
    general, neuron, connections, network_layout, \
        external_source = load_config(config_path)

    ## folder for results
    PATH_TO_OUTPUT = os.path.join(
                    'results',
                    f"ed_{network_layout['energy_dependent']}",
                    time.strftime("%Y_%m_%d_%H%M%S")+f"_seed_{general['seed']}")
    if general['record']:
        create_folder(PATH_TO_OUTPUT)
        # create figure folder
        PATH_TO_FIGS = os.path.join(PATH_TO_OUTPUT, 'figures')
        create_folder(PATH_TO_FIGS)
        # create data folder
        PATH_TO_DATA = os.path.join(PATH_TO_OUTPUT, 'data')
        create_folder(PATH_TO_DATA)
        # save configuration file
        save_config(config_path, PATH_TO_OUTPUT)
