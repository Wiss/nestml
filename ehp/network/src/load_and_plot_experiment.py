"""
load_and_plot_experiments.py
    load results from experiments and regenerater plots
"""
import time
import os

from src.utils.manage_files import (create_folder,
                                    load_config,
                                    save_data,
                                    load_data)

from src.utils.measurement_tools import (get_weight_matrix,
                                         get_adjacency_matrix,
                                         get_graph_measurement,
                                         get_clustering_coeff,
                                         get_mean_energy_per_neuron,
                                         energy_fix_point)

from src.utils.figures import (create_weights_figs,
                               create_spikes_figs,
                               create_pops_figs,
                               create_multimeter_figs,
                               create_matrices_figs,
                               create_full_matrix_figs,
                               create_graph_measure_figs,
                               create_cc_vs_incoming_figs,
                               create_cc_vs_atp_figs,
                               delays_hist,
                               weights_before_after_hist)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='load and plot experiment',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", type=str, required=True,
            help="Folder path \
            relative to results\ folder")
    args = parser.parse_args()
    config_path = args.path
    # count time
    start_sim = time.time()

    PATH_TO_CONFIG = os.path.join(
                    'results',
                    config_path)
    if os.path.exists(PATH_TO_CONFIG):
        print("Oh! Now I remember. I know exactly what you mean.")
    else:
        print("No sir, we are vey sorry to inform that there is no backup for such expeirment")

    PATH_TO_FIGS = os.path.join(
                    PATH_TO_CONFIG,
                    time.strftime("%Y_%m_%d_%H%M%S"))
    create_folder(PATH_TO_FIGS)
    PATH_TO_DATA = os.path.join(
                    PATH_TO_CONFIG,
                    'data')

    general, neurons, connections, network_layout, \
        external_sources = load_config('results/' + config_path + '/config.yaml')

    # load data
    pop_dict = load_data(PATH_TO_DATA, 'pop_dict')
    adj_matrix_fin = load_data(PATH_TO_DATA, 'adjacency_matrix_fin')
    adj_matrix_init = load_data(PATH_TO_DATA, 'adjacency_matrix_init')
    degree_ex = {}
    degree_in = {}
    degree_ex['init'] = load_data(PATH_TO_DATA, 'degree_measurement_ex_init')
    degree_ex['fin'] = load_data(PATH_TO_DATA, 'degree_measurement_ex_fin')
    degree_in['init'] = load_data(PATH_TO_DATA, 'degree_measurement_in_init')
    degree_in['fin'] = load_data(PATH_TO_DATA, 'degree_measurement_in_fin')
    spikes_events = load_data(PATH_TO_DATA, 'spikes')
    multimeter_events = load_data(PATH_TO_DATA, 'multimeter')
    strength_ex = {}
    strength_in = {}
    strength_ex['init'] = load_data(PATH_TO_DATA, 'strength_measurement_ex_init')
    strength_ex['fin'] = load_data(PATH_TO_DATA, 'strength_measurement_ex_fin')
    strength_in['init'] = load_data(PATH_TO_DATA, 'strength_measurement_in_init')
    strength_in['fin'] = load_data(PATH_TO_DATA, 'strength_measurement_in_fin')
    w_matrix_fin = load_data(PATH_TO_DATA, 'weight_matrix_fin')
    w_matrix_init = load_data(PATH_TO_DATA, 'weight_matrix_init')
    weights_fin = load_data(PATH_TO_DATA, 'weights_fin')
    weights_init = load_data(PATH_TO_DATA, 'weights_init')
    full_w_matrix_fin = load_data(PATH_TO_DATA, 'weight_matrix_full_fin')
    full_w_matrix_init = load_data(PATH_TO_DATA, 'weight_matrix_full_init')
    full_adj_matrix_fin = load_data(PATH_TO_DATA, 'adjacency_matrix_full_fin')
    full_adj_matrix_init = load_data(PATH_TO_DATA, 'adjacency_matrix_full_init')
    clustering_coeff_init = load_data(PATH_TO_DATA, 'clustering_coeff_init')
    clustering_coeff_fin = load_data(PATH_TO_DATA, 'clustering_coeff_fin')

    assert all(degree_ex['init']['in'] == get_graph_measurement(matrices=adj_matrix_init,
                                                               pop='ex')['in'])

    # choose simulation subsection
    init_time = 0
    fin_time = 120

    if init_time >= fin_time:
        raise Exception("init time should be smaller than fin time")

    pops_figs = 0
    spikes_figs = 1
    matrices_figs = 0
    full_matrix_figs = 0
    graph_measure_figs = 0
    delays_hist_figs = 0
    multimeter_figs = 1
    cc_vs_incoming_figs = 0

    # generate plots
    # position plots
    if pops_figs:
        create_pops_figs(pop=pop_dict,
                         fig_name="pop_positions",
                         output_path=PATH_TO_FIGS)

    if spikes_figs:
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
                        time_window=general['firing_rate_window'],
                        init_time=init_time,
                        fin_time=fin_time,
                        record_rate=general['record_rate']
                        )

    ## Matrices
    # plot w_matrices
    # for every 'matrix type' save it inm the <matrices> dict with
    # <type>_matrix_<when>
    # <type> could be: weight, adjacency
    # <when> could be: init and fin
    if matrices_figs:
        matrices = {'weight_matrix_init': w_matrix_init,
                    'weight_matrix_fin': w_matrix_fin,
                    'adjacency_matrix_init': adj_matrix_init,
                    'adjacency_matrix_fin': adj_matrix_fin
                    }
        for matrix_k, matrix_v in matrices.items():
            # save
            matrix_name = matrix_k.split('_')[0].capitalize()
            matrix_str = matrix_k.split('_')[1]
            title = matrix_name + ' ' + matrix_str
            create_matrices_figs(matrix=matrix_v,
                                output_path=PATH_TO_FIGS,
                                fig_name=matrix_k,
                                title=title)
    if full_matrix_figs:
        full_matrices = {'weight_matrix_full_init': full_w_matrix_init,
                            'weight_matrix_full_fin': full_w_matrix_fin,
                            'adjacency_matrix_full_init': full_adj_matrix_init,
                            'adjacency_matrix_full_fin': full_adj_matrix_fin
                            }
        for f_matrix_k, f_matrix_v in full_matrices.items():
            create_full_matrix_figs(matrix=f_matrix_v,
                                    output_path=PATH_TO_FIGS,
                                    fig_name=f_matrix_k,
                                    title=f_matrix_k)

    # plot histrograms
    # for every 'measurement type' save it in the <measurement> dict with
    # <type>_measurement_<when>
    # <type> could be: strengh, degree
    # <when> could be: init and fin
    if graph_measure_figs:
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
            # plot
            splited_key = measure_k.split('_')
            pop_pre = splited_key[2]
            if pop_pre == 'ex':
                pop_pre = 'excitatory'
            elif pop_pre == 'in':
                pop_pre = 'inhibitory'
            measure = splited_key[0]
            title = f'Histogram: {pop_pre} population {measure}'
            create_graph_measure_figs(measure=measure_v,
                                        output_path=PATH_TO_FIGS,
                                        fig_name=measure_k,
                                        title=title,
                                        cumulative=False)
            # create plots with cumulative distribution
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

    if delays_hist_figs:
        # plot delays histograms
        delays_hist(weights_init=weights_init,
                    output_path=PATH_TO_FIGS)

    if cc_vs_incoming_figs:
        # in-degree(strength) vs out-degree(strength) vs clustering coeff
        incoming_clustering_dict = {'1':
            {'name': 'strenght_init',
            'cc': clustering_coeff_init,
            'matrix': full_w_matrix_init},
                                    '2':
            {'name': 'degree_init',
            'cc': clustering_coeff_init,
            'matrix': full_adj_matrix_init},
                                    '3':
            {'name': 'strenght_fin',
            'cc': clustering_coeff_fin,
            'matrix': full_w_matrix_fin},
                                    '4':
            {'name': 'degree_fin',
            'cc': clustering_coeff_fin,
            'matrix': full_adj_matrix_fin}
                                    }
        for key, value in incoming_clustering_dict.items():
            for n in range(2):
                if n == 0:
                    pop = 'all'
                    pop_length = pop_dict['ex']['n'] + pop_dict['in']['n']
                else:
                    pop = 'ex'
                    pop_length = pop_dict['ex']['n']
                incoming_var = value['name'].split('_')[0]
                when = value['name'].split('_')[-1]
                create_cc_vs_incoming_figs(clustering_coeff=value['cc'],
                                        matrix=value['matrix'],
                                        incoming_var=incoming_var,
                                        population=pop,
                                        pop_length=pop_length,
                                        fig_name= when + '_' + pop,
                                        output_path=PATH_TO_FIGS)

        # in-degree(strength) vs out-degree(strength) vs mean energy per neuron
        mean_energy_per_neuron = get_mean_energy_per_neuron(
                                            ATP=multimeter_events)
        incoming_energy_dict = {'1':
            {'name': 'strenght_fin',
            'a': mean_energy_per_neuron,
            'matrix': full_w_matrix_fin},
                                    '2':
            {'name': 'degree_fin',
            'a': mean_energy_per_neuron,
            'matrix': full_adj_matrix_fin}
                                    }
        for key, value in incoming_energy_dict.items():
            for n in range(2):
                if n == 0:
                    pop = 'all'
                    pop_length = pop_dict['ex']['n'] + pop_dict['in']['n']
                else:
                    pop = 'ex'
                    pop_length = pop_dict['ex']['n']
                incoming_var = value['name'].split('_')[0]
                when = value['name'].split('_')[-1]
                create_cc_vs_incoming_figs(clustering_coeff=value['a'],
                                        matrix=value['matrix'],
                                        incoming_var=incoming_var,
                                        population=pop,
                                        pop_length=pop_length,
                                        fig_name= 'ATP_' + when + '_' + pop,
                                        cc_var='<ATP>',
                                        output_path=PATH_TO_FIGS)

        for n in range(2):
            if n == 0:
                pop = 'all'
                pop_length = pop_dict['ex']['n'] + pop_dict['in']['n']
            else:
                pop = 'ex'
                pop_length = pop_dict['ex']['n']
            create_cc_vs_atp_figs(clustering_coeff=clustering_coeff_fin,
                                  mean_atp=mean_energy_per_neuron,
                                  population=pop,
                                  pop_length=pop_length,
                                  fig_name= '_fin_' + pop,
                                  output_path=PATH_TO_FIGS)
    if multimeter_figs:
        for measurement in general['record']['multimeter']:
            create_multimeter_figs(multimeter_events=multimeter_events,
                                measurement=measurement,
                                fig_name=measurement,
                                output_path=PATH_TO_FIGS,
                                simtime=general['simtime'],
                                init_time=init_time,
                                fin_time=fin_time,
                                multimeter_record_rate=general['record_rate'])

    end_all = time.time()
    plots_tot_time = -(start_sim - end_all)
    print('##########################################')
    print('## COMPUTE TIMES #########################')
    print('##########################################')
    print(f'plots takes: {plots_tot_time} sec. -> {plots_tot_time/60} min.')
