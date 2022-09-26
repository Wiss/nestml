import numpy as np

from src.logging.logging import logger
logger = logger.getChild(__name__)

def instantaneous_phase(pop, activity: dict, final_t: float,
                      resolution: float, **kargs) -> dict:
    """
    calculate instantaneous phase

    Parameters
    ----------
    pop:
        population
    activity:
        dictinary with senders and time information
    final_t:
        simulaton final time
    resolution:
        simulation resolution in ms

    Returns
    -------
    instantaneous_phase:
        dictionary with senders and ins_phase information
    """
    kargs.setdefault('verbose', 0)
    step = resolution
    instantaneous_phase = {}
    last_spike_time = {}
    instantaneous_phase['times'] = np.arange(start=0,
                                             stop=final_t,
                                             step=step)
    for neuron in pop.tolist():
        instantaneous_phase.setdefault(neuron,
                                        np.empty(
                                        len(instantaneous_phase['times'])
                                        ))
        # init a nan array for each neuron in pop population
        instantaneous_phase[neuron].fill(np.nan)
        # for recording last time that sender had an spike
        last_spike_time.setdefault(neuron, 0)
        if kargs['verbose'] > 0:
            print('neuron in i_phase')
            print(neuron)
    if kargs['verbose'] > 0:
        print('instantaneous_phase')
        print(instantaneous_phase)

    #for sender in set(activity['senders']):
    #    instantaneous_phase.setdefault(sender,
    #                                   np.empty(
    #                                       len(instantaneous_phase['times'])
    #                                   ))
    #    instantaneous_phase[sender].fill(np.nan)
        # for recording last time that sender had an spike
    #   last_spike_time.setdefault(sender, 0)

    # only update neurons present in activity[senders]
    for spike_time, sender in zip(activity['times'], activity['senders']):
        if kargs['verbose'] > 0:
            print('sender')
            print(sender)
        if last_spike_time[sender] == 0:
            pass
        else:
            start_idx_from_last_spk_time = int(last_spike_time[sender] /
                                             step)
            end_idx_from_spk_time = int(spike_time / step)
            for idx in range(start_idx_from_last_spk_time,
                          end_idx_from_spk_time):
                t_from_idx = idx * step
                instantaneous_phase[sender][idx] = 2 * np.pi * (t_from_idx -
                                last_spike_time[sender]) / (spike_time -
                                last_spike_time[sender])
        last_spike_time[sender] = spike_time
    return instantaneous_phase

def population_avg_order_param(i_phase_pop: dict, **kargs) -> dict:
    """
    calculate population average order parameter

    Parameter
    ---------
    i_phase_pop:
        instantaneous phase for pop population

    Returns
    -------
    pop_avg_order_param:
        population average order parameter
    """
    kargs.setdefault('verbose', 0)
    i_phase_pop_copy = i_phase_pop.copy()
    pop_avg_order_param = {}
    pop_avg_order_param['times'] = i_phase_pop_copy['times']
    pop_avg_order_param['o_param'] = np.zeros(len(i_phase_pop['times'])) * 1j
    del[i_phase_pop_copy['times']]
    neuron_count = 0
    for neuron in i_phase_pop_copy:
        # add both arrays treating NaNs as zeros
        pop_avg_order_param['o_param'] = np.nansum(
                                            np.dstack((
                                            pop_avg_order_param['o_param'],
                                            np.e ** (1j *
                                            i_phase_pop_copy[neuron])
                                            )), axis=2)
        neuron_count += 1
    if kargs['verbose'] > 0:
        print('neuron count in pop_avg_order_param')
        print(neuron_count)
    return pop_avg_order_param, neuron_count

def phase_coherence(pop_dict: dict, spikes_events: dict, final_t: float,
                  resolution: float, **kargs) -> (dict, dict):
    """
    calculates phase coherence

    Parameters
    ----------
    pop_dict:
       dictionary weith populations
    spikes_events:
        spikes_events[population]['senders' or 'times']
    final_t:
        simulations final time (this is total time,
        assuming we start at time zero)
    resolution:
        simulation resolution in ms

    Returns
    -------
    phase_coherence:
        per population (key ex or inh) an for the whole network (key all)
    """
    kargs.setdefault('verbose', 0)
    i_phase = {}
    pop_average_order_param = {}
    #global_inst_order_param = {}
    neuron_count = {'all': 0}
    #for pop, activity in spikes_events.items():
    for pop in pop_dict:
        activity = spikes_events[pop]
        i_phase[pop] = instantaneous_phase(pop=pop_dict[pop],
                                           activity=activity,
                                           final_t=final_t,
                                           resolution=resolution)
        pop_average_order_param[pop], neuron_count[pop] = \
                                            population_avg_order_param(
                                                i_phase_pop=i_phase[pop])
        if kargs['verbose'] > 0:
            print(f'pop_average_order_param for {pop} pop')
            print(neuron_count[pop])
        # set default values for 'all' case
        pop_average_order_param.setdefault('all', {})
        pop_average_order_param['all'].setdefault('o_param',
                                    np.zeros((1, len(i_phase[pop]['times']))) * 1j)
        pop_average_order_param['all']['o_param'] += \
                                    pop_average_order_param[pop]['o_param']
        neuron_count['all'] += neuron_count[pop]
        # obtain fasor lenght (phase coherence) per pop
        pop_average_order_param[pop]['o_param'] = abs(
                                    pop_average_order_param[pop]['o_param'] /
                                    neuron_count[pop])
    pop_average_order_param['all']['times'] = \
                                    pop_average_order_param['ex']['times']
    pop_average_order_param['all']['o_param'] = abs(
                                    pop_average_order_param['all']['o_param'] /
                                    neuron_count['all'])
    if kargs['verbose'] > 0:
        print('total neurons inside phase coherence')
        print(neuron_count)
        #global_inst_order_param[pop] = global_inst_order_param()
    return i_phase, pop_average_order_param

def pop_firing_rate(pop_dict: dict, spikes_events: dict, time_window: int,
                  final_t: float, resolution: float, **kargs) -> dict:
    """
    calculates population average firing rate

    Parameters:
    -----------
    pop_dict:
        dictionary with populations
    spikes_events:
        dicitonary with spikes information
    time_window:
        size of time window calculate average firing rate in ms
    final_t:
        final simulation time in ms
    resolution:
        simulation resolution in ms
    """
    kargs.setdefault('verbose', 0)
    step = resolution
    idx_from_time_window = int(time_window / step)
    firing_rate = {}
    firing_rate['times'] = np.arange(start=0,
                                     stop=final_t + step,
                                     step=step)

    for pop in pop_dict:
        firing_rate.setdefault(pop, {})
        firing_rate[pop].setdefault('tot_spikes',
                                    np.zeros(len(firing_rate['times'])))
        firing_rate[pop].setdefault('n_neurons',
                                    len(pop_dict[pop]))
        for spike_time in spikes_events[pop]['times']:
            idx_from_spk_time = int(spike_time / step)
            firing_rate[pop]['tot_spikes'][idx_from_spk_time] += 1

    # calculate spike events for all neurons together
    for pop in pop_dict:
        firing_rate.setdefault('all', {})
        firing_rate['all'].setdefault('tot_spikes',
                                      np.zeros(len(firing_rate['times'])))
        firing_rate['all'].setdefault('n_neurons', 0)
        firing_rate['all']['tot_spikes'] += firing_rate[pop]['tot_spikes']
        firing_rate['all']['n_neurons'] += firing_rate[pop]['n_neurons']

    # calculate rates
    for key in ['ex', 'in', 'all']:
        if kargs['verbose'] > 0:
            print(f'n_neurons for {key} key')
            print(firing_rate[key]['n_neurons'])
            print('len firing rate times')
            print(len(firing_rate['times']))
            print('zeros with len(finirg_rate[times])')
            print(np.zeros(len(firing_rate['times'])))
        firing_rate[key].setdefault('rates',
                               np.zeros(len(firing_rate['times'])))
        if kargs['verbose'] > 0:
            print(f'firing_rates[{key}][rates]')
            print(firing_rate[key]['rates'])
            print(f'firing_rate[{key}]')
            print(firing_rate[key])
        for idx in range(len(firing_rate['times'])):
            if idx <= idx_from_time_window:
                firing_rate[key]['rates'][idx] = \
                    sum(firing_rate[key]['tot_spikes'][0:idx])
            else:
                firing_rate[key]['rates'][idx] = \
                    sum(
                    firing_rate[key]['tot_spikes'][idx-idx_from_time_window:idx]
                        )
        if kargs['verbose'] > 0:
            print('time_window')
            print(time_window)
            print('firing rate')
            print(firing_rate)
            print(f'firing_rate[{key}]')
            print(firing_rate[key])
            print(f'firing_rate[{key}][rates]')
            print(firing_rate[key]['rates'])
        firing_rate[key]['rates'] *= 1000 / time_window / firing_rate[key]['n_neurons']
    return firing_rate

def get_weight_matrix(pop: dict, weights: dict, **kargs) -> (dict, np.array):
    """
    return weight matrix with

    Parameters
    ----------
    pop:
        dictionary with population
    weights:
        weights dictionary

    Returns
    -------
    weight_matrix:
        weight matrix for possible connections between population's present
        in pop
        M_{ij} represents connection from neuron i to neuron j
    """

    w_matrix = {}
    # init weight matrix with zeros
    for n, con_key in enumerate(weights):
        pre = con_key.split('_')[0]
        post = con_key.split('_')[1]
        w_matrix[con_key] = np.zeros([len(pop[pre]), len(pop[post])])
        min_pre_pop_idx = min(pop[pre].tolist())
        min_post_pop_idx = min(pop[post].tolist())
        weights_source = weights[con_key]['source']
        weights_target = weights[con_key]['target']
        if any(item in weights[con_key].get('synapse_model')[0].split('_') \
               for item in ['edlif', 'rec', 'copy']):
            w_key = 'w'
        else:
            w_key = 'weight'
        # fill w_matrix
        for idx in range(len(weights[con_key][w_key])):
            # re-number axes to start with idx==0 instead of 'source' values
            w_matrix[con_key][weights_source[idx] - min_pre_pop_idx,
                              weights_target[idx] - min_post_pop_idx] += \
                                  weights[con_key][w_key][idx]
    return w_matrix

def get_adjacency_matrix(weight_matrix: dict, threshold: float,
                       **kargs) -> dict:
    """
    given weight_matrix it returns adjacency matrix

    Parameters
    ----------
    weigth_matirx:
        weight matrix dictionary
    trheshold:
        threhold to calculate a_ij = w_ij * [w_ij> threshold]

    Returns
    -------
    adj_matrix: dict
        dictionary with adjacency matices
    """
    adj_matrix = {}
    kargs.setdefault('verbose', 0)
    for w_k, w_v in weight_matrix.items():
        adj_matrix[w_k] = (abs(weight_matrix[w_k]) > threshold) * 1
        if kargs['verbose'] > 0:
            print('adj_matrix[w_k] type')
            print(type(adj_matrix[w_k]))
            if kargs['verbose'] > 1:
                print('w_v')
                print(w_v)
                print('weight_matrix')
                print(weight_matrix[w_k])
                print('adj_matrix')
                print(adj_matrix[w_k])
    return adj_matrix

def get_graph_measurement(matrices: dict, pop: str, **kargs) -> dict:
    """
    given matrix (weight, adjacency) return strengh or degree, respectively

    Parameters
    ----------
    matrices:
        dictionary associated with matrices between populations.
        M_{ij} represent connection from neuron i to neuron j
    pop:
        string indicating over which population the measurement is calculated
    """
    kargs.setdefault('verbose', 0)
    measured = {}
    neuron_i = {}
    n, m = 0, 0
    for matrix_k, matrix_v in matrices.items():
        pop_pre = matrix_k.split('_')[0]
        pop_post = matrix_k.split('_')[1]
        matrix_v_wo_diag = matrix_v
        if pop_pre == pop_post:
            # take out diagonal if we are analyzing the same pre and post pops
            np.fill_diagonal(matrix_v_wo_diag, 0)
        else:
            pass
            # matrix_v_wo_diag = matrix_v
        if pop_pre == pop:
            # from pop_pre I can get outgoing connections
            # K_{pre_i}^{out} = \sum_j A_{ij}
            if n == 0:
                neuron_i['out'] = np.sum(matrix_v_wo_diag, axis=1)*0
            neuron_i['out'] += np.sum(matrix_v_wo_diag, axis=1)
            n += 1
            if kargs['verbose'] > 0:
                print('pop pre')
                print(pop_pre)
                print('neuron_in')
                print(neuron_i['in'])
        if pop_post == pop:
                # from pop_post I can get incoming connections
                # K_{post_i}^{in} = \sum_j A_{ji}
            if m == 0:
                neuron_i['in'] = np.sum(matrix_v_wo_diag, axis=0)*0
            neuron_i['in'] += np.sum(matrix_v_wo_diag, axis=0)
            m += 1
            if kargs['verbose'] > 0:
                print('pop_post')
                print(pop_post)
                print('neuron_out')
                print(neuron_i['out'])
    return neuron_i
