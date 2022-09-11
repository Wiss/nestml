"""
experiment.py
    perform one experiment using the biophysical network and the yaml config
    file in config
"""
import time
import os

from src.logging.logging import logger
import src.network as network
from src.utils.figures import (create_weights_figs,
                               create_spikes_figs,
                               create_graph_figs,
                               create_pops_figs,
                               create_multimeter_figs)
from src.utils.manage_files import (create_folder,
                                    load_config,
                                    save_config,
                                    save_data,
                                    load_data)

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
    if general['record']['spikes'] or general['record']['weights']:
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
    pop_dict, conn_dict, weight_rec_dict = network.init_network(
                                            seed=general['seed'],
                                            neurons=neurons,
                                            connections=connections,
                                            network_layout=network_layout,
                                            external_source=external_source)
    print(pop_dict['ex'])
    print(pop_dict['ex'].spatial)
    print(pop_dict['in'])
    print(pop_dict['in'].spatial)
    print(pop_dict['in'].spatial["positions"])
    #plt.show()
    print(conn_dict['ex_ex'])
    create_pops_figs(pop=pop_dict,
                     fig_name="pop_positions",
                     output_path=PATH_TO_FIGS)

    # run network
    logger.info("running network")
    spikes, multimeter, weights = network.run_network(
                                            simtime=general['simtime'],
                                            record=general['record'],
                                            record_rate=general['record_rate'],
                                            pop_dict=pop_dict,
                                            weight_rec_dict=weight_rec_dict)

    logger.info("simulation finished successfully")

    # save data
    rec_dict = {'spikes': spikes,
                'multimeter': multimeter,
                'weights': weights}
    for rec_k, rec_dict_v in rec_dict.items():
        data = {}
        if general['record'][rec_k]:
            for key, value in rec_dict_v.items():
                if value is None:
                    data[key] = None
                else:
                    data[key] = value.get('events')
            # get events for figures
            if rec_k == 'spikes':
                spikes_events = data
                create_spikes_figs(spikes_events=spikes_events,
                                   fig_name='spikes',
                                   output_path=PATH_TO_FIGS)
            elif rec_k == 'multimeter':
                multimeter_events = data
            if rec_k == 'weights':
                weights_events = data
            save_data(PATH_TO_DATA, rec_k, data)
            logger.info("recordable %s recorded", rec_k)

    # generate figs (only if data is recorded)
    #for rec_k, rec_dict_v in rec_dict.items():
    #    if general['record'][rec_k]:
    #        for key, value in rec_dict_v.items():
    #        PATH_TO_FIGS
