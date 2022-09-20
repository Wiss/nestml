#import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

import src.utils.measurement_tools as tools

# parameters
alpha = 0.6
fontsize_title = 14
fontsize_label = 12
fontsize_legend = 9
linewidth = 2
pointsize = 20
fig_size = (12, 12)

#from pynestml.frontend.pynestml_frontend import generate_nest_target
#NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")


def create_weights_figs(weights_events: dict, fig_name: str, output_path: str,
                      **kargs):
    kargs.setdefault('hlines', 0)
    kargs.setdefault('cont_lines', 0)
    kargs.setdefault('legend', 0)
    kargs.setdefault('verbose', 0)
    # This should be here, it's just a test
    for key, val in weights_events.items():
        if val is not None:
            w = {}
            valid_pair = []
            for source, target, time, weight in zip(weights_events['ex_ex']['senders'],
                                                    weights_events['ex_ex']['targets'],
                                                    weights_events['ex_ex']['times'],
                                                    weights_events['ex_ex']['weights']):

                valid_pair.append([source, target])
                if kargs['verbose']:
                    print('source')
                    print(source)
                    print('target')
                    print(target)
                    print('time')
                    print(time)
                    print('weight')
                    print(weight)

                w.setdefault(str(source), {})
                w[str(source)].setdefault(str(target), {})
                w[str(source)][str(target)].setdefault('times', []).append(time)
                w[str(source)][str(target)].setdefault('weights', []).append(weight)

            # do not include repeated valid pairs
            valid_pair_set = [i for n, i in enumerate(valid_pair) if i not in valid_pair[:n]]
            color = iter(cm.rainbow(np.linspace(0, 1, len(valid_pair_set))))
            fig, ax = plt.subplots(1, figsize=fig_size)
            ax.set_title(f'weights {key.split("_")[0]} -> {key.split("_")[1]}',
                            fontsize=fontsize_title)
            for s, t in valid_pair_set:
                c = next(color)
                if kargs['verbose']:
                    print(f'source {s}, target {t}')
                ax.plot(w[str(s)][str(t)]['times'], w[str(s)][str(t)]['weights'], '.',
                        color=c, label=f'{s}->{t}')
                prev_time = 0.
                last_time = kargs['simtime']
                if kargs['hlines']:
                    for n, weight in enumerate(w[str(s)][str(t)]['weights']):
                        time = w[str(s)][str(t)]['times'][n]
                        if kargs['verbose']:
                            print(f'weight {weight}')
                            print(f'time {time}')
                        if n < len(w[str(s)][str(t)]['weights']) - 1:
                            next_time = w[str(s)][str(t)]['times'][n+1]
                            ax.hlines(y=weight, xmin=time, xmax=next_time, color=c)
                        else:
                            ax.hlines(y=weight, xmin=time, xmax=last_time, color=c)
                        prev_time = time
                if kargs['cont_lines']:
                    ax.plot(w[str(s)][str(t)]['times'], w[str(s)][str(t)]['weights'],
                            '--', color=c)
            ax.set_ylabel('Weights TODO:units', fontsize=fontsize_label)
            ax.set_xlabel('Time (ms)', fontsize=fontsize_label)
            if kargs['legend']:
                ax.legend(fontsize=fontsize_legend)
            save_weights_fig =f'{output_path}/{fig_name}_{key}'
            plt.savefig(save_weights_fig, dpi=500)

def create_spikes_figs(spikes_events: dict, multimeter_events: dict,
                     fig_name: str, output_path: str, **kargs):
    """
    Raster plots

    Parameters
    ----------
    spikes_events:
        dictionary with all spike's information (events)
    multimeter_events:
        dictionary with all multimeters information
    fig_name:
        figure name
    output_path:
        path to figure
    kargs:
        extra parameters to the funcion given as dictionary
    """
    kargs.setdefault('mult_var', None)
    kargs.setdefault('alpha', 0.5)
    kargs.setdefault('mean_lw', 3)
    final_t = kargs['simtime']
    mult_time = np.arange(start=kargs['multimeter_record_rate'], stop=final_t,
                          step=kargs['multimeter_record_rate'])
    fig, ax = plt.subplots(2*len(spikes_events), figsize=fig_size, sharex=True)
    ax[-1].set_xlabel('time (ms)', fontsize=fontsize_label)
    n = 0
    for pop in spikes_events:
        if pop == 'ex':
            color = 'r'
            p = 'excitatory'
        elif pop == 'in':
            color = 'b'
            p = 'inhibitory'
        senders = spikes_events[pop]['senders']
        times = spikes_events[pop]['times']
        ax[n].plot(times, senders, '.', c=color)
        ax[n].set_title('Spikes from ' + p + ' population',
                        fontsize=fontsize_title)
        ax[n].set_ylabel('Neuron ID', fontsize=fontsize_label)

        ax[n+1].set_ylabel('R (syncronization)', fontsize=fontsize_label)
        n += 2
    # save image
    save_spikes_s_fig =f'{output_path}/{fig_name}_separate'
    plt.savefig(save_spikes_s_fig, dpi=500)


    # all together
    fig, ax = plt.subplots(3, figsize=fig_size, sharex=True)
    ax[0].set_title('Spikes', fontsize=fontsize_title)
    ax[1].set_title('Firing rate and energy', fontsize=fontsize_title)
    ax[2].set_title('Synchronization: Population avergae order parameter',
                    fontsize=fontsize_title)
    ax[-1].set_xlabel('time (ms)', fontsize=fontsize_label)
    for pop, events in spikes_events.items():
        if pop == 'ex':
            color = 'r'
            p = 'excitatory'
        elif pop == 'in':
            color = 'b'
            p = 'inhibitory'
        senders = spikes_events[pop]['senders']
        times = spikes_events[pop]['times']
        atp_per_sender = {}
        for sender, atp in zip(multimeter_events[pop]['senders'],
                            multimeter_events[pop]['ATP']):
            atp_per_sender.setdefault(str(sender), []).append(atp)
        # spikes
        ax[0].plot(times, senders, '.', c=color, label=pop)
        # ATP
        for n, sender in enumerate(atp_per_sender):
            if n == 0:
                ax[1].plot(mult_time, atp_per_sender[sender], '.', c=color,
                           label=pop, alpha=kargs['alpha'])
                ax[1].plot(mult_time, atp_per_sender[sender], c=color,
                           alpha=kargs['alpha'])
            else:
                ax[1].plot(mult_time, atp_per_sender[sender], '.', c=color,
                           alpha=kargs['alpha'])
                ax[1].plot(mult_time, atp_per_sender[sender], c=color,
                           alpha=kargs['alpha'])
        atp_total = [sum(t_atp) for t_atp in zip(*list(atp_per_sender.values()))]
        ax[1].plot(mult_time, [atp_t/len(atp_per_sender) for atp_t in atp_total],
                   c=color, label=pop + '_mean', lw=kargs['mean_lw'])
        # phase coherence
        phase_coherence = tools.phase_coherence(spikes_events=spikes_events,
                                                final_t=final_t)
        ax[2].plot(phase_coherence[pop]['times'],
                phase_coherence[pop]['o_param'],
                   c=color, label=pop, lw=kargs['mean_lw'],
                   alpha=alpha)

    ## phase coherence for all
    n_neurons_ex = int(kargs['n_neurons'] * kargs['ex_in_ratio'])
    n_neurons_in = kargs['n_neurons'] - n_neurons_ex
    all_phase_coherence = (phase_coherence['ex']['o_param'] * n_neurons_ex  +
                           phase_coherence['in']['o_param'] * n_neurons_in) / \
                           (n_neurons_in + n_neurons_ex)
    #ax[2].plot(phase_coherence['ex']['times'],
    #           all_phase_coherence,
    #           c='k', label='all', lw=kargs['mean_lw'], alpha=alpha)

    ax[0].set_ylabel('Neuron ID', fontsize=fontsize_label)
    ax[0].legend(fontsize=fontsize_legend)
    ax[1].set_ylabel('ATP (%)', fontsize=fontsize_label)
    ax[1].legend(fontsize=fontsize_legend)
    ax[2].set_ylabel('R', fontsize=fontsize_label)
    ax[2].legend(fontsize=fontsize_legend)
    # save image
    save_spikes_j_fig =f'{output_path}/{fig_name}_joint'
    plt.savefig(save_spikes_j_fig, dpi=500)

def create_pops_figs(pop: dict, fig_name: str, output_path: str, **kargs):
    # all together
    fig, ax = plt.subplots(1, figsize=fig_size, sharex=True)
    ax.set_title('Neurons positions')
    for key in pop.keys():
        if key == 'ex':
            color = 'r'
        elif key == 'in':
            color = 'b'
        pos = pop[key].spatial['positions']
        for n, p in enumerate(pos):
            if n == 0:
                ax.plot(p[0], p[1], '.', c=color, label=key,
                        markersize=pointsize)
            else:
                ax.plot(p[0], p[1], '.', c=color, markersize=pointsize)
    ax.set_ylabel('y (mm)', fontsize=fontsize_label)
    ax.set_xlabel('x (mm)', fontsize=fontsize_label)
    ax.legend(fontsize=fontsize_legend)
    # save image
    save_spikes_j_fig =f'{output_path}/{fig_name}_joint'
    plt.savefig(save_spikes_j_fig, dpi=500)

def create_multimeter_figs(multimeter_events: dict, fig_name: str,
                         output_path: str, **kargs):
    pass

def create_graph_figs():
    pass
