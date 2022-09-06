"""
experiment.py
    perform one experiment using the biophysical network and the yaml config
    file in config
"""
#import datetime
import time
#import yaml
import os
#import copy

from src.logging.logging import logger
import src.network as network
#from utils.figures import create_weights_figs, \
#create_spikes_figs, create_graph_figs
from src.utils.manage_files import create_folder, load_config, save_config, \
    save_data, load_data
#from utils.bionet_tools import weight_init, population_pos_init

from src.logging.logging import logger


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
    general, neurons, connections, network_layout, \
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

    # setup network
    logger.info("setting up network")
    pop_ex, pop_in = network.init_network(neurons=neurons,
                                          connections=connections,
                                          network_layout=network_layout,
                                          external_source=external_source)
    print(pop_ex)
    print(pop_ex.spatial)
    print(pop_in)
    print(pop_in.spatial)

    # run network
    logger.info("running network")
    results = network.run_network(simtime=general['simtime'],
                                  seed=general['seed'],
                                  record=general['record'],
                                  record_rate=general['record_rate'])

    logger.info("simulation finished successfully")
