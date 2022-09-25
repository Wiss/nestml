"""
experiment.py
    perform one experiment using the biophysical network and the yaml config
    file in config
"""
import copy
import os
import subprocess
import time

from src.logging.logging import logger
import src.network as network
from src.utils.figures import (create_weights_figs,
                               create_spikes_figs,
                               create_pops_figs,
                               create_multimeter_figs,
                               create_matrices_figs,
                               create_graph_measure_figs,
                               delays_hist,
                               weights_before_after_hist)
from src.utils.manage_files import (create_folder,
                                    load_config,
                                    save_config,
                                    save_data,
                                    load_data)
from src.utils.measurement_tools import (get_weight_matrix,
                                         get_adjacency_matrix,
                                         get_graph_measurement)

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
        external_sources = load_config(config_path)

    ## folder for results
    g_m_ex = neurons['ex']['params']['energy_params']['gamma']['mean']
    g_s_ex = neurons['ex']['params']['energy_params']['gamma']['std']
    g_m_in = neurons['in']['params']['energy_params']['gamma']['mean']
    g_s_in = neurons['in']['params']['energy_params']['gamma']['std']
    PATH_TO_OUTPUT = os.path.join(
                    'results',
                    f"ed_{network_layout['energy_dependent']}_" \
                    f"gex_m_{g_m_ex}_s_{g_s_ex}_" \
                    f"gin_m_{g_m_in}_s_{g_s_in}",
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
    pop_dict, conn_dict, weight_rec_dict, external_srcs, \
                            subregion_mults = network.init_network(
                                            resolution=general['resolution'],
                                            module=general['module'],
                                            seed=general['seed'],
                                            neurons=neurons,
                                            connections=connections,
                                            network_layout=network_layout,
                                            external_sources=external_sources)

    # run network
    logger.info("running network")
    spikes, multimeter, weights, weights_init, weights_fin = \
                                            network.run_network(
                                            simtime=general['simtime'],
                                            record=general['record'],
                                            record_rate=general['record_rate'],
                                            pop_dict=pop_dict,
                                            conn_dict=conn_dict,
                                            weight_rec_dict=weight_rec_dict)

    logger.info("simulation finished successfully")

    # save data and generate plots
    # position plots
    create_pops_figs(pop=pop_dict,
                     fig_name="pop_positions",
                     output_path=PATH_TO_FIGS)
    # spikes, multimeter and weights data
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
                spikes_events = copy.deepcopy(data)
            elif rec_k == 'multimeter':
                multimeter_events = copy.deepcopy(data)
            if rec_k == 'weights':
                weights_events = copy.deepcopy(data)
                create_weights_figs(weights_events=weights_events,
                                    fig_name='weights',
                                    output_path=PATH_TO_FIGS,
                                    simtime=general['simtime'],
                                    hlines=True,
                                    cont_lines=True,
                                    legend=False)
            save_data(PATH_TO_DATA, rec_k, data)
            logger.info("recordable %s saved", rec_k)

    # TODO include recording condition
    create_spikes_figs(pop_dict=pop_dict,
                       spikes_events=spikes_events,
                       multimeter_events=multimeter_events,
                       fig_name='spikes',
                       output_path=PATH_TO_FIGS,
                       mult_var=general['record']['multimeter'],
                       alpha=0.2,
                       multimeter_record_rate=general['record_rate'],
                       simtime=general['simtime'],
                       resolution=general['resolution'],
                       n_neurons=network_layout['n_neurons'],
                       ex_in_ratio=network_layout['ex_in_ratio'],
                       time_window=general['firing_rate_window'])

    # record inital weights
    save_data(PATH_TO_DATA, 'weights_init', weights_init)
    logger.info("recordable weights_init saved")
    # record final weights
    save_data(PATH_TO_DATA, 'weights_fin', weights_fin)
    logger.info("recordable weights_fin saved")

    ## Matrices
    # get and record init weight maxtrices
    w_matrix_init = get_weight_matrix(pop=pop_dict,
                                      weights=weights_init)
    # get and record init weight maxtrices
    w_matrix_fin = get_weight_matrix(pop=pop_dict,
                                     weights=weights_fin)
    # plot w_matrices
    # for every 'matrix type' save it inm the <matrices> dict with
    # <type>_matrix_<when>
    # <type> could be: weight, adjacency
    # <when> could be: init and fin
    adj_threshold = general['adj_threshold']
    adj_matrix_init = get_adjacency_matrix(
                           weight_matrix=w_matrix_init,
                           threshold=adj_threshold)
    adj_matrix_fin = get_adjacency_matrix(
                                 weight_matrix=w_matrix_fin,
                                 threshold=adj_threshold)
    matrices = {'weight_matrix_init': w_matrix_init,
                'weight_matrix_fin': w_matrix_fin,
                'adjacency_matrix_init': adj_matrix_init,
                'adjacency_matrix_fin': adj_matrix_fin
                }
    for matrix_k, matrix_v in matrices.items():
        # save
        save_data(PATH_TO_DATA, matrix_k, matrix_v)
        logger.info("recordable '%s' saved", matrix_k)
        # plot = matrix_k.split('_')[0].capitalize()
        matrix_name = matrix_k.split('_')[0].capitalize()
        matrix_str = matrix_k.split('_')[1]
        title = matrix_name + ' ' + matrix_str
        logger.info('ploting %s figure', title)
        create_matrices_figs(matrix=matrix_v,
                                   output_path=PATH_TO_FIGS,
                                   fig_name=matrix_k,
                                   title=title)

    # plot histrograms
    # for every 'measurement type' save it in the <measurement> dict with
    # <type>_measurement_<when>
    # <type> could be: strengh, degree
    # <when> could be: init and fin
    strength_ex = {}
    strength_in = {}
    degree_ex = {}
    degree_in = {}
    #measurement_list = [strenght_ex, strenght_in, degree_ex, degree_in]
    #matrix_list = [w_matrix_init, stre, degree_ex, degree_in]
   # for measurement, matrix in zip(measurement_list, matrix_lisi):
   #     for pop_k in ['ex', 'in']
   #         for when_k in ['init', 'fin']:
   #             measurement[when_k] = get_graph_measurement(
    #                                            matrices=w_matrix_init,
    #                                           pop='ex')
    strength_ex['init'] = get_graph_measurement(matrices=w_matrix_init,
                                                pop='ex')
    strength_ex['fin'] = get_graph_measurement(matrices=w_matrix_fin,
                                               pop='ex')
    strength_in['init'] = get_graph_measurement(matrices=w_matrix_init,
                                                pop='in')
    strength_in['fin'] = get_graph_measurement(matrices=w_matrix_fin,
                                               pop='in')
    degree_ex['init'] = get_graph_measurement(matrices=adj_matrix_init,
                                              pop='ex')
    degree_ex['fin'] = get_graph_measurement(matrices=adj_matrix_fin,
                                             pop='ex')
    degree_in['init'] = get_graph_measurement(matrices=adj_matrix_init,
                                              pop='in')
    degree_in['fin'] = get_graph_measurement(matrices=adj_matrix_fin,
                                             pop='in')
    measurements = {'strength_measurement_ex_init': strength_ex['init'],
                    'strength_measurement_ex_fin': strength_ex['fin'],
                    'strength_measurement_in_init': strength_in['init'],
                    'strength_measurement_in_fin': strength_in['fin'],
                    'degree_measurement_ex_init': degree_ex['init'],
                    'degree_measurement_ex_fin': degree_ex['fin'],
                    'degree_measurement_in_init': degree_in['init'],
                    'degree_measurement_in_fin': degree_in['fin'],
                    }
    for measure_k, measure_v in measurements.items():
        # save
        save_data(PATH_TO_DATA, measure_k, measure_v)
        logger.info("recordable '%s' saved", measure_k)
        # plot
        splited_key = measure_k.split('_')
        pop_pre = splited_key[2]
        if pop_pre == 'ex':
            pop_pre = 'excitatory'
        elif pop_pre == 'in':
            pop_pre = 'inhibitory'
        measure = splited_key[0]
        title = f'Histogram: {pop_pre} population {measure}'
        logger.info('ploting %s figure', title)
        create_graph_measure_figs(measure=measure_v,
                                  output_path=PATH_TO_FIGS,
                                  fig_name=measure_k,
                                  title=title,
                                  cumulative=False)
        if measure == 'degree':
            # only for degree bc strenght has negative values
            create_graph_measure_figs(measure=measure_v,
                                    output_path=PATH_TO_FIGS,
                                    fig_name=measure_k,
                                    title=title,
                                    logscale=True,
                                    cumulative=-1)

    # plot init and fin weights histograms
    weights_before_after_hist(weights_init=weights_init,
                              weights_fin=weights_fin,
                              output_path=PATH_TO_FIGS)

    # plot delays histograms
    delays_hist(weights_init=weights_init,
                output_path=PATH_TO_FIGS)


    # save logger into experiment folder
    subprocess.run(['cp', 'src/last_experiment.log', f'{PATH_TO_OUTPUT}'])

    # generate figs (only if data is recorded)
    #for rec_k, rec_dict_v in rec_dict.items():
    #    if general['record'][rec_k]:
    #        for key, value in rec_dict_v.items():
    #        PATH_TO_FIGS
