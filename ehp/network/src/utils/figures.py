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
            for source, target, time, weight in zip(weights_events[key]['senders'],
                                                    weights_events[key]['targets'],
                                                    weights_events[key]['times'],
                                                    weights_events[key]['weights']):

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
            ax.set_ylabel('Weights', fontsize=fontsize_label)
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
    kargs.setdefault('plot_i_phase', False)
    kargs.setdefault('time_window', 30)
    kargs.setdefault('alpha', 0.3)
    kargs.setdefault('mean_lw', 3)
    kargs.setdefault('plot_pop_p_coherence_with_all', False)
    kargs.setdefault('plot_each_n_atp', False)
    final_t = kargs['simtime']
    atp_stck = {}
    atp_stack = {}
    atp_mean = {}
    atp_std = {}
    mult_time = np.arange(start=kargs['multimeter_record_rate'], stop=final_t,
                          step=kargs['multimeter_record_rate'])
    firing_rate = tools.pop_firing_rate(spikes_events=spikes_events,
                                        time_window=kargs['time_window'],
                                        final_t=final_t)
    i_phase, phase_coherence = tools.phase_coherence(
                                            spikes_events=spikes_events,
                                            final_t=final_t)
    for pop in spikes_events:
        fig, ax = plt.subplots(3, figsize=fig_size, sharex=True)
        ax[-1].set_xlabel('time (ms)', fontsize=fontsize_label)
        if pop == 'ex':
            color = 'darkred'
            p = 'excitatory'
        elif pop == 'in':
            color = 'steelblue'
            p = 'inhibitory'
        senders = spikes_events[pop]['senders']
        times = spikes_events[pop]['times']
        ax[0].set_title('Spikes from ' + p + ' population',
                        fontsize=fontsize_title)
        ax[1].set_title('Available energy and firing rate from ' + p + \
                        ' population', fontsize=fontsize_title)
        ax[2].set_title('Synchronization: Phase coherence from ' + p + \
                        ' population',
                        fontsize=fontsize_title)
        ax[0].set_ylabel('Neuron ID', fontsize=fontsize_label)
        ax[1].set_ylabel('ATP (%)', fontsize=fontsize_label,
                         color='darkgreen')
        ax[2].set_ylabel('R', fontsize=fontsize_label)
        # spikes
        ax[0].plot(times, senders, '.', c=color)
        # ATP and firing rate
        atp_per_sender = {}
        for sender, atp in zip(multimeter_events[pop]['senders'],
                            multimeter_events[pop]['ATP']):
            atp_per_sender.setdefault(str(sender), []).append(atp)
        atp_stck[pop] = [np.array(atp_s) for atp_s in atp_per_sender.values()]
        atp_stack[pop] = np.stack(atp_stck[pop])
        atp_mean[pop] = np.mean(atp_stack[pop], axis=0)
        atp_std[pop] = np.std(atp_stack[pop], axis=0)
        ax[1].plot(mult_time,
                   atp_mean[pop],
                   c='darkgreen',
                   lw=kargs['mean_lw'],
                   label=pop + ' mean')
        ax[1].fill_between(mult_time,
                           atp_mean[pop] - atp_std[pop],
                           atp_mean[pop] + atp_std[pop],
                           edgecolor='darkgreen',
                           color='darkgreen',
                           label=pop + ' sd',
                           alpha=kargs['alpha'])
        ax[1].legend(fontsize=fontsize_legend)
        # second axes for firing rate
        ax_1_2 = ax[1].twinx()
        ax_1_2.plot(firing_rate['times'],
                    firing_rate[pop]['rates'],
                    c='darkorange',
                    lw=kargs['mean_lw'],
                    label=pop + ' fr',
                    alpha=0.5)
        ax_1_2.set_ylabel('Firing rate (Hz)',
                          fontsize=fontsize_legend,
                          color='darkorange')
        #ax_1_2.legend(fontsize=fontsize_legend)
        # phase coherence
        ax[2].set_ylabel('R',
                         fontsize=fontsize_label)
        ax[2].plot(phase_coherence[pop]['times'],
                   phase_coherence[pop]['o_param'].reshape(-1),
                   c='darkgrey', label=pop, lw=kargs['mean_lw'])
        if kargs['plot_i_phase']:
            for sender in i_phase[pop]:
                if sender != 'times':
                    ax[2].plot(i_phase[pop]['times'],
                            i_phase[pop][sender],
                            c=color, label=pop + 'inst_phase',
                            lw=kargs['mean_lw'], alpha=alpha)
        # save image
        save_spikes_s_fig =f'{output_path}/{fig_name}_{pop}_separate'
        plt.savefig(save_spikes_s_fig, dpi=500)


    # all together
    fig, ax = plt.subplots(3, figsize=fig_size, sharex=True)
    ax[0].set_title('Spikes', fontsize=fontsize_title)
    ax[1].set_title('Available energy and firing rate',
                    fontsize=fontsize_title)
    ax[2].set_title('Synchronization: Phase coherence',
                    fontsize=fontsize_title)
    ax[-1].set_xlabel('time (ms)', fontsize=fontsize_label)
    for pop, events in spikes_events.items():
        if pop == 'ex':
            color = 'darkred'
            p = 'excitatory'
        elif pop == 'in':
            color = 'steelblue'
            p = 'inhibitory'
        senders = spikes_events[pop]['senders']
        times = spikes_events[pop]['times']
        # spikes
        ax[0].plot(times, senders, '.', c=color, label=pop)
        # ATP
        if kargs['plot_each_n_atp']:
            for n, sender in enumerate(atp_per_sender):
                if n == 0:
                    ax[1].plot(mult_time, atp_per_sender[sender],
                               '.', c=color, label=pop, alpha=kargs['alpha'])
                    ax[1].plot(mult_time, atp_per_sender[sender], c=color,
                            alpha=kargs['alpha'])
                else:
                    ax[1].plot(mult_time, atp_per_sender[sender],
                               '.', c=color, alpha=kargs['alpha'])
                    ax[1].plot(mult_time, atp_per_sender[sender], c=color,
                            alpha=kargs['alpha'])
        # phase coherence
        if kargs['plot_pop_p_coherence_with_all']:
            ax[2].plot(phase_coherence[pop]['times'],
                    phase_coherence[pop]['o_param'].reshape(-1),
                    c='darkgrey', label=pop, lw=kargs['mean_lw'],
                    alpha=alpha)
        if kargs['plot_i_phase']:
            for sender in i_phase[pop]:
                if sender != 'times':
                    ax[2].plot(i_phase[pop]['times'],
                               i_phase[pop][sender],
                               c=color, label=pop + 'inst_phase',
                               lw=kargs['mean_lw'], alpha=alpha)

    ## phase coherence for all
    #n_neurons_ex = int(kargs['n_neurons'] * kargs['ex_in_ratio'])
    #n_neurons_in = kargs['n_neurons'] - n_neurons_ex
    #all_phase_coherence = (phase_coherence['ex']['o_param'] * n_neurons_ex  +
    #                       phase_coherence['in']['o_param'] * n_neurons_in) / \
    #                       (n_neurons_in + n_neurons_ex)
    # ATP
    atp_stack['all'] = np.stack(atp_stck['ex'] + atp_stck['in'])
    atp_mean['all'] = np.mean(atp_stack['all'], axis=0)
    atp_std['all'] = np.std(atp_stack['all'], axis=0)
    ax[1].plot(mult_time,
                atp_mean['all'],
                c='darkgreen',
                lw=kargs['mean_lw'],
                label='mean')
    ax[1].fill_between(mult_time,
                        atp_mean['all'] - atp_std['all'],
                        atp_mean['all'] + atp_std['all'],
                        edgecolor='darkgreen',
                        color='darkgreen',
                        label='sd',
                        alpha=kargs['alpha'])
    # second axes for firing rate
    ax_1_2 = ax[1].twinx()
    ax_1_2.plot(firing_rate['times'],
                firing_rate['all']['rates'],
                c='darkorange',
                lw=kargs['mean_lw'],
                label='fr',
                alpha=0.5)
    ax_1_2.set_ylabel('Firing rate (Hz)',
                      fontsize=fontsize_legend,
                      color='darkorange')
    #ax_1_2.legend(fontsize=fontsize_legend)
    # phase coherence
    ax[2].plot(phase_coherence['all']['times'],
               phase_coherence['all']['o_param'].reshape(-1),
               c='darkgrey', lw=kargs['mean_lw'])  # , alpha=alpha)

    ax[0].set_ylabel('Neuron ID', fontsize=fontsize_label)
    ax[0].legend(fontsize=fontsize_legend)
    ax[1].set_ylabel('ATP (%)', fontsize=fontsize_label,
                     color='darkgreen')
    ax[1].legend(fontsize=fontsize_legend)
    ax[2].set_ylabel('R', fontsize=fontsize_label)
    #ax[2].legend(fontsize=fontsize_legend)
    # save image
    save_spikes_j_fig =f'{output_path}/{fig_name}_joint'
    plt.savefig(save_spikes_j_fig, dpi=500)

def create_pops_figs(pop: dict, fig_name: str, output_path: str, **kargs):
    # all together
    fig, ax = plt.subplots(1, figsize=fig_size, sharex=True)
    ax.set_title('Neurons positions')
    for key in pop.keys():
        if key == 'ex':
            color = 'darkred'
        elif key == 'in':
            color = 'steelblue'
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


def weights_before_after_hist(weights_init: dict, weights_fin: dict,
                            output_path: str, **kargs):
    """
    for each key in weights_init and weight_fin plot weight values histogram

    Parameters
    ----------
    weights_init:
        initial weights before simulation
    weight_fin
        final weights after simulation
    """
    kargs.setdefault('facecolor', 'steelblue')
    kargs.setdefault('n_bins', 20)
    kargs.setdefault('rwidth', 0.9)
    for w_k, w_v in weights_init.items():
        fig, ax = plt.subplots(2, figsize=fig_size, sharex=True,
                               sharey=True)
        fig.suptitle(f'Histogram: {w_k} connections with ' \
                     f'{w_v.get("synapse_model")[0]} synapse type',
                     fontsize=fontsize_title)
        for axes in ax:
            axes.set_ylabel('Frequency', fontsize=fontsize_label)
            axes.set_xlabel('Weight', fontsize=fontsize_label)
        if 'rec' in w_v.get("synapse_model")[0].split('_'):
            # the condition above is note the proper one, because we really
            # want to check if the synapse model is an energy_dependent (ed)
            # one or not
            w_key = 'w'
        else:
            w_key = 'weight'
        ax[0].set_title('Before',
                        fontsize=fontsize_title)
        ax[1].set_title('After',
                        fontsize=fontsize_title)
        ax[0].hist(weights_init[w_k][w_key], bins=kargs['n_bins'],
                   facecolor=kargs['facecolor'],
                   rwidth=kargs['rwidth'])
        ax[1].hist(weights_fin[w_k][w_key], bins=kargs['n_bins'],
                   facecolor=kargs['facecolor'],
                   rwidth=kargs['rwidth'])
        # save image
        save_weights_fig =f'{output_path}/{w_k}_before_after_hist'
        plt.savefig(save_weights_fig, dpi=500)
