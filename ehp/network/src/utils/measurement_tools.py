import numpy as np

def instantaneous_phase(activity: dict, final_t: float) -> dict:
    """
    calculate instantaneous phase

    Parameters
    ----------
    activity:
        dictinary with senders and time information
    final_t:
        simulaton final time

    Returns
    -------
    instantaneous_phase:
        dictionary with senders and ins_phase information
    """
    step = 0.1
    # TODO fix step = 0.1, this value should come from the config file
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
    pop_avg_order_param['o_param'] = abs(pop_avg_order_param['o_param'] /
                                         n_senders)
    #pop_avg_order_param['o_param'].reshape(-1)
    return pop_avg_order_param

def phase_coherence(spikes_events: dict, final_t: float) -> (dict, dict):
    """
    calculates phase coherence

    Parameters
    ----------
    spikes_events:
        spikes_events[population]['senders' or 'times']
    final_t:
        simulations final time (this is total time,
        assuming we start at time zero)

    Returns
    -------
    phase_coherence:
        per population (key ex or inh) an for the whole network (key all)
    """
    i_phase = {}
    pop_average_order_param = {}
    global_inst_order_param = {}
    for pop, activity in spikes_events.items():
        i_phase[pop] = instantaneous_phase(activity=activity,
                                           final_t=final_t)
        pop_average_order_param[pop] = population_avg_order_param(
                                                i_phase_pop=i_phase[pop])

        #global_inst_order_param[pop] = global_inst_order_param()
    return i_phase, pop_average_order_param
