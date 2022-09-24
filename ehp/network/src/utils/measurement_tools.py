import numpy as np

def instantaneous_phase(activity: dict, final_t: float,
                      resolution: float) -> dict:
    """
    calculate instantaneous phase

    Parameters
    ----------
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
    step = resolution
    instantaneous_phase = {}
    last_spike_time = {}
    instantaneous_phase['times'] = np.arange(start=0,
                                             stop=final_t,
                                             step=step)
    for sender in set(activity['senders']):
        instantaneous_phase.setdefault(sender,
                                       np.empty(
                                           len(instantaneous_phase['times'])
                                       ))
        instantaneous_phase[sender].fill(np.nan)
        # for recording last time that sender had an spike
        last_spike_time.setdefault(sender, 0)

    for spike_time, sender in zip(activity['times'], activity['senders']):
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

def population_avg_order_param(i_phase_pop: dict) -> dict:
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
    i_phase_pop_copy = i_phase_pop.copy()
    pop_avg_order_param = {}
    pop_avg_order_param['times'] = i_phase_pop_copy['times']
    pop_avg_order_param['o_param'] = np.zeros(len(i_phase_pop['times'])) * 1j
    del[i_phase_pop_copy['times']]
    n_senders = 0
    for sender in i_phase_pop_copy:
        # add both arrays treating NaNs as zeros
        pop_avg_order_param['o_param'] = np.nansum(
                                            np.dstack((
                                            pop_avg_order_param['o_param'],
                                            np.e ** (1j *
                                            i_phase_pop_copy[sender])
                                            )), axis=2)
        n_senders += 1
    return pop_avg_order_param, n_senders

def phase_coherence(spikes_events: dict, final_t: float,
                  resolution:float) -> (dict, dict):
    """
    calculates phase coherence

    Parameters
    ----------
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
    i_phase = {}
    pop_average_order_param = {}
    #global_inst_order_param = {}
    n_senders = {'all': 0}
    for pop, activity in spikes_events.items():
        i_phase[pop] = instantaneous_phase(activity=activity,
                                           final_t=final_t,
                                           resolution=resolution)
        pop_average_order_param[pop], n_senders[pop] = \
                                            population_avg_order_param(
                                                i_phase_pop=i_phase[pop])
        # set default values for 'all' case
        pop_average_order_param.setdefault('all', {})
        pop_average_order_param['all'].setdefault('o_param',
                                    np.zeros((1, len(i_phase[pop]['times']))) * 1j)
        pop_average_order_param['all']['o_param'] += \
                                    pop_average_order_param[pop]['o_param']
        n_senders['all'] += n_senders[pop]
        # obtain fasor lenght (phase coherence)
        pop_average_order_param[pop]['o_param'] = abs(
                                    pop_average_order_param[pop]['o_param'] /
                                         n_senders[pop])
    pop_average_order_param['all']['times'] = \
                                    pop_average_order_param['ex']['times']
    pop_average_order_param['all']['o_param'] = abs(
                                    pop_average_order_param['all']['o_param'] /
                                    n_senders['all'])
        #global_inst_order_param[pop] = global_inst_order_param()
    return i_phase, pop_average_order_param

def pop_firing_rate(spikes_events: dict, time_window: int, final_t: float,
                  resolution: float) -> dict:
    """
    calculates population average firing rate

    Parameters:
    -----------
    spikes_events:
        dicitonary with spikes information
    time_window:
        size of time window calculate average firing rate in ms
    final_t:
        final simulation time in ms
    resolution:
        simulation resolution in ms
    """
    step = resolution
    idx_from_time_window = int(time_window / step)
    firing_rate = {}
    firing_rate['times'] = np.arange(start=0,
                                     stop=final_t + step,
                                     step=step)

    for pop in spikes_events:
        firing_rate.setdefault(pop, {})
        firing_rate[pop].setdefault('tot_spikes',
                                    np.zeros(len(firing_rate['times'])))
        firing_rate[pop].setdefault('n_neurons',
                                    len(set(spikes_events[pop]['senders'])))
        for spike_time in spikes_events[pop]['times']:
            idx_from_spk_time = int(spike_time / step)
            firing_rate[pop]['tot_spikes'][idx_from_spk_time] += 1
            #firing_rate['all']['tot_spikes'][idx_from_spk_time] += 1

    # calculate spike events for all neurons together
    for pop in spikes_events:
        firing_rate.setdefault('all', {})
        firing_rate['all'].setdefault('tot_spikes',
                                      np.zeros(len(firing_rate['times'])))
        firing_rate['all'].setdefault('n_neurons', 0)
        firing_rate['all']['tot_spikes'] += firing_rate[pop]['tot_spikes']
        firing_rate['all']['n_neurons'] += firing_rate[pop]['n_neurons']

    # calculate rates
    for key in ['ex', 'in', 'all']:
        firing_rate[key].setdefault('rates',
                               np.zeros(len(firing_rate['times'])))
        for idx in range(len(firing_rate['times'])):
            if idx <= idx_from_time_window:
                firing_rate[key]['rates'][idx] = \
                    sum(firing_rate[key]['tot_spikes'][0:idx])
            else:
                firing_rate[key]['rates'][idx] = \
                    sum(
                    firing_rate[key]['tot_spikes'][idx-idx_from_time_window:idx]
                        )
        firing_rate[key]['rates'] *= 1000 / time_window / firing_rate[key]['n_neurons']
    return firing_rate

def get_weight_matrix(pop: dict, weights: dict) -> dict:
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
        threhold to calculate a_ij = w_ij * [w_ij> thresold]

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
        dictionary associated with matrices between populations
    pop:
        string indicating over which population the measurement is calculated
    """
    kargs.setdefault('verbose', 1)
    measured = {}
    neuron_i = {}
    n, m = 0, 0
    for matrix_k, matrix_v in matrices.items():
        pop_pre = matrix_k.split('_')[0]
        pop_post = matrix_k.split('_')[1]
        print('pop_pre')
        print(pop_pre)
        print('pop_post')
        print(pop_post)
        if pop_pre == pop:
            if n == 0:
                neuron_i['in'] = np.sum(matrix_v, axis=1)*0
            neuron_i['in'] += np.sum(matrix_v, axis=1)
            n += 1
            if kargs['verbose'] > 0:
                print('pop pre')
                print(pop_pre)
                print('neuron_in')
                print(neuron_i['in'])
        if pop_post == pop:
            if m == 0:
                neuron_i['out'] = np.sum(matrix_v, axis=0)*0
            neuron_i['out'] += np.sum(matrix_v, axis=0)
            m += 1
            if kargs['verbose'] > 0:
                print('pop_post')
                print(pop_post)
                print('neuron_out')
                print(neuron_i['out'])
    return neuron_i
