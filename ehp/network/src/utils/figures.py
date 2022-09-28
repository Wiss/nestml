#import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from src.logging.logging import logger
import src.utils.measurement_tools as tools


logger = logger.getChild(__name__)

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
            plt.close(fig)

def create_spikes_figs(pop_dict: dict, spikes_events: dict,
                     multimeter_events: dict, fig_name: str,
                     output_path: str, **kargs):
    """
    Raster plots

    Parameters
    ----------
    pop_dict:
        dictionary with populations
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
    resolution = kargs['resolution']
    atp_stck = {}
    atp_stack = {}
    atp_mean = {}
    atp_std = {}
    mult_time = np.arange(start=kargs['multimeter_record_rate'], stop=final_t,
                          step=kargs['multimeter_record_rate'])
    firing_rate = tools.pop_firing_rate(pop_dict=pop_dict,
                                        spikes_events=spikes_events,
                                        time_window=kargs['time_window'],
                                        final_t=final_t,
                                        resolution=resolution)
    i_phase, phase_coherence = tools.phase_coherence(
                                            pop_dict=pop_dict,
                                            spikes_events=spikes_events,
                                            final_t=final_t,
                                            resolution=resolution)
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
        ax[0].grid(axis='x')
        ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].grid(axis='x')
        ax[2].grid(axis='x')
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
        logger.info('%s population ATP average through simulation %s',
                    pop, np.mean(atp_mean[pop]))
        # second axes for firing rate
        ax_1_2 = ax[1].twinx()
        ax_1_2.plot(firing_rate['times'],
                    firing_rate[pop]['rates'],
                    c='darkorange',
                    lw=kargs['mean_lw'],
                    label=pop + ' fr',
                    alpha=0.5)
        ax_1_2.set_ylabel('Firing rate (Hz)',
                          fontsize=fontsize_label,
                          color='darkorange')
        logger.info('%s population Firing rate average through simulation %s Hz',
                    pop, np.mean(firing_rate[pop]['rates']))
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
        logger.info('%s population phase coherence average through simulation %s',
                    pop, np.nanmean(phase_coherence[pop]['o_param']))
        # save image
        save_spikes_s_fig =f'{output_path}/{fig_name}_{pop}_separate'
        plt.savefig(save_spikes_s_fig, dpi=500)
        plt.close(fig)


    # all together
    fig, ax = plt.subplots(3, figsize=fig_size, sharex=True)
    ax[0].set_title('Spikes', fontsize=fontsize_title)
    ax[1].set_title('Available energy and firing rate',
                    fontsize=fontsize_title)
    ax[2].set_title('Synchronization: Phase coherence',
                    fontsize=fontsize_title)
    ax[-1].set_xlabel('time (ms)', fontsize=fontsize_label)
    ax[0].grid(axis='x')
    ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid(axis='x')
    ax[2].grid(axis='x')
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
    logger.info('all populations ATP average through simulation %s',
                np.mean(atp_mean['all']))
    # second axes for firing rate
    ax_1_2 = ax[1].twinx()
    ax_1_2.plot(firing_rate['times'],
                firing_rate['all']['rates'],
                c='darkorange',
                lw=kargs['mean_lw'],
                label='fr',
                alpha=0.5)
    ax_1_2.set_ylabel('Firing rate (Hz)',
                      fontsize=fontsize_label,
                      color='darkorange')
    logger.info('all population Firing rate average through simulation %s Hz',
                np.mean(firing_rate['all']['rates']))
    #ax_1_2.legend(fontsize=fontsize_legend)
    # phase coherence
    ax[2].plot(phase_coherence['all']['times'],
               phase_coherence['all']['o_param'].reshape(-1),
               c='darkgrey', lw=kargs['mean_lw'])  # , alpha=alpha)
    logger.info('all population phase coherence average through simulation %s',
                np.nanmean(phase_coherence['all']['o_param']))

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
    plt.close(fig)

def create_pops_figs(pop: dict, fig_name: str, output_path: str, **kargs):
    # all together
    fig, ax = plt.subplots(1, figsize=fig_size, sharex=True)
    ax.set_title('Neurons positions', fontsize=fontsize_title)
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
    plt.close(fig)

def create_multimeter_figs(multimeter_events: dict, measurement: str,
                         fig_name: str, output_path: str, **kargs):
    final_t = kargs['simtime']
    #resolution = kargs['resolution']
    mult_time = np.arange(start=kargs['multimeter_record_rate'], stop=final_t,
                          step=kargs['multimeter_record_rate'])
    for pop in multimeter_events:
        if pop == 'ex':
            p = 'excitatory'
        elif pop == 'in':
            p = 'inhibitory'
        measure_per_sender = {}
        fig, ax = plt.subplots(1, figsize=fig_size)
        ax.set_title(measurement + ' in ' + p + ' population',
                     fontsize=fontsize_title)
        ax.set_xlabel('time (ms)', fontsize=fontsize_label)
        ax.set_ylabel(measurement, fontsize=fontsize_label)
        ax.grid(axis='x')
        for sender, measure in zip(multimeter_events[pop]['senders'],
                            multimeter_events[pop][measurement]):
            measure_per_sender.setdefault(sender, []).append(measure)
        for sender in set(multimeter_events[pop]['senders']):
            ax.plot(mult_time,
                    measure_per_sender[sender],
                    #c=f'C{n}',
                    alpha=0.5,
                    label=sender)
        #measure_stck[pop] = [np.array(measure_s) for measure_s
        #                    in measure_per_sender.values()]
        #measure_stack[pop] = np.stack(measure_stck[pop])
        #measure_mean[pop] = np.mean(measure_stack[pop], axis=0)
        #measure_std[pop] = np.std(measure_stack[pop], axis=0)
        #ax[1].plot(mult_time,
        #           measure_per_sender[pop],
        #           c='darkgreen',
        #           lw=kargs['mean_lw'],
        #           label=pop + ' mean')
        ax.legend(fontsize=fontsize_legend)
        # save image
        save_measurement_fig =f'{output_path}/{fig_name}_{pop}'
        plt.savefig(save_measurement_fig, dpi=500)
        plt.close(fig)

def create_graph_measure_figs(measure: dict, output_path: str, **kargs):
    """
    create graph measurement figures


    Parameters
    ----------
    measure:
        measurement informations
    """
    kargs.setdefault('density', True)
    kargs.setdefault('cumulative', -1) # True -> cumulative, -1 reversed cumulative
    kargs.setdefault('histtype', 'bar')
    kargs.setdefault('logscale', False)
    kargs.setdefault('facecolor', 'steelblue')
    kargs.setdefault('n_bins', 50)
    kargs.setdefault('rwidth', 0.9)
    if kargs['logscale'] and not kargs['cumulative']:
        # only plot logscale if disitrbution is cumulative
        raise ValueError(f'if logscale is {kargs["logscale"]}, ' \
                         'then "cumulative" must be "True"')
    measuremnt = kargs['fig_name'].split('_')[0]
    if measuremnt == 'strength':
        symbol = 'S'
    elif measuremnt == 'degree':
        symbol = 'K'
    if kargs['cumulative'] in [True, -1]:
        eq_sym = '>'
        kargs['histtype'] = 'step'
        kargs['n_bins'] = 50
        cum = True
    else:
        eq_sym = '='
        cum = False
    if kargs['density']:
        y_label = f'P({measuremnt}{eq_sym}{symbol})'
        kargs['rwidth'] = 1
    else:
        y_label = 'Frequency'

    for m_k, w_v in measure.items():
        fig, ax = plt.subplots(2, figsize=fig_size, sharex=True,
                               sharey=True)
        fig.suptitle(f'{kargs["title"]}', fontsize=fontsize_title)
        for axes in ax:
            axes.set_ylabel(y_label, fontsize=fontsize_label)
            axes.set_xlabel(f'{symbol}', fontsize=fontsize_label)
        ax[0].set_title(f'In-{measuremnt}',
                        fontsize=fontsize_title)
        ax[1].set_title(f'Out-{measuremnt}',
                        fontsize=fontsize_title)
        _hist_in, bins_in = np.histogram(measure['in'],
                                         bins=kargs['n_bins'])
        _hist_out, bins_out = np.histogram(measure['out'],
                                           bins=kargs['n_bins'])
        # a little dirty. Could be improve!
        if kargs['logscale']:
            logbins_in = np.logspace(np.log10(bins_in[0]),
                                     np.log10(bins_in[-1]),
                                     len(bins_in))
            logbins_out = np.logspace(np.log10(bins_out[0]),
                                      np.log10(bins_out[-1]),
                                      len(bins_out))
        else:
            logbins_in = bins_in
            logbins_out = bins_out
        ax[0].hist(measure['in'], bins=logbins_in,
                   facecolor=kargs['facecolor'],
                   rwidth=kargs['rwidth'],
                   density=kargs['density'],
                   cumulative=kargs['cumulative'],
                   histtype=kargs['histtype'])
        ax[1].hist(measure['out'], bins=logbins_out,
                   facecolor=kargs['facecolor'],
                   rwidth=kargs['rwidth'],
                   density=kargs['density'],
                   cumulative=kargs['cumulative'],
                   histtype=kargs['histtype'])
        if kargs['logscale']:
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
        # save image
        save_measurement_fig =f'{output_path}/{kargs["fig_name"]}_cum_{cum}'
        plt.savefig(save_measurement_fig, dpi=500)
        plt.close(fig)


def create_matrices_figs(matrix: dict, output_path: str, **kargs):
    """
    plot matrices (weight, strengh and degree) for each connection
    between populations
    """
    kargs.setdefault('pad', '5%')
    kargs.setdefault('cmap', 'jet')  # jet
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(kargs['title'], fontsize=fontsize_title)
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[:-1, :-1])
    ax2 = plt.subplot(gs[:-1, -1])
    ax3 = plt.subplot(gs[-1, :-1])
    ax4 = plt.subplot(gs[-1, -1])
    ax_list = [ax1, ax2, ax3, ax4]
    for n, key in enumerate(matrix):
        ax = ax_list[n]
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes('right', size="5%", pad=kargs['pad'])
        fig = ax.get_figure()
        fig.add_axes(ax_cb)
        w = matrix[key]
        im = ax.imshow(w)
        im.set_cmap(kargs['cmap'])
        plt.colorbar(im, cax=ax_cb)
        ax.set_title(f'{key.split("_")[0]} -> {key.split("_")[1]}')
        plt.tight_layout()
        ax_cb.yaxis.set_tick_params(labelright=True)

    # save image
    save_weights_m_fig =f'{output_path}/{kargs["fig_name"]}'
    plt.savefig(save_weights_m_fig, pad_inches=0, dpi=500)
    plt.close(fig)


def create_full_matrix_figs(matrix: np.array, output_path: str, **kargs):
    """
    plot full matrix
    """
    kargs.setdefault('pad', '5%')
    kargs.setdefault('cmap', 'jet')  # jet
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(kargs['title'], fontsize=fontsize_title)
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size="5%", pad=kargs['pad'])
    fig.add_axes(ax_cb)
    w = matrix
    im = ax.imshow(w)
    im.set_cmap(kargs['cmap'])
    plt.colorbar(im, cax=ax_cb)
    plt.tight_layout()
    ax_cb.yaxis.set_tick_params(labelright=True)

    # save image
    save_weights_f_m_fig =f'{output_path}/{kargs["fig_name"]}'
    plt.savefig(save_weights_f_m_fig, dpi=500)
    plt.close(fig)

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
        if any(item in w_v.get('synapse_model')[0].split('_') \
               for item in ['edlif', 'rec', 'copy']):
            w_key = 'w'
        else:
            w_key = 'weight'
        ax[0].set_title('Initial',
                        fontsize=fontsize_title)
        ax[1].set_title('Final',
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
        plt.close(fig)


def delays_hist(weights_init, output_path: str, **kargs):
    """
    for each key in weights_init or and weight_fin plot weight values histogram

    Parameters
    ----------
    weights_init:
        initial or final weights
    """
    kargs.setdefault('facecolor', 'steelblue')
    kargs.setdefault('n_bins', 20)
    kargs.setdefault('rwidth', 0.9)
    for w_k, w_v in weights_init.items():
        fig, ax = plt.subplots(1, figsize=fig_size, sharex=True,
                               sharey=True)
        fig.suptitle(f'Histogram: {w_k} connections with ' \
                     f'{w_v.get("synapse_model")[0]} synapse type',
                     fontsize=fontsize_title)
        ax.set_ylabel('Frequency', fontsize=fontsize_label)
        ax.set_xlabel('Delays', fontsize=fontsize_label)
        if any(item in w_v.get('synapse_model')[0].split('_') \
               for item in ['edlif', 'rec', 'copy']):
            w_key = 'delay'
        else:
            w_key = 'delay'
        ax.set_title('Distribution of delays',
                        fontsize=fontsize_title)
        ax.hist(weights_init[w_k][w_key], bins=kargs['n_bins'],
                   facecolor=kargs['facecolor'],
                   rwidth=kargs['rwidth'])
        # save image
        save_weights_fig =f'{output_path}/{w_k}_delays_hist'
        plt.savefig(save_weights_fig, dpi=500)
        plt.close(fig)


def create_cc_vs_incoming_figs(clustering_coeff: np.array,
                             matrix: np.array,
                             incoming_var: str,
                             output_path: str, **kargs):
    """
    Plot clustering coefficient vs incoming strength ot degree

    Parameters
    ----------
    matrix:
        matrix
    """
    kargs.setdefault('markersize', 2)
    kargs.setdefault('markerfacecolor', 'steelblue')
    kargs.setdefault('edgecolors', 'k')
    kargs.setdefault('lw', None)
    kargs.setdefault('cmap', 'gray_r')
    kargs.setdefault('size', 60)
    kargs.setdefault('fig_size', (12, 12))
    #cmap_pos = [0.95, 0.5, 0.05, 0.3]
    incoming = []
    outgoing = []
    both = []
    for idx, _ in enumerate(clustering_coeff):
        incoming.append(np.sum(matrix[idx, :]))
        outgoing.append(np.sum(matrix[:, idx]))
        both.append(incoming[-1] + outgoing[-1])

    # in vs out histogram
    fig, ax = plt.subplots(figsize=kargs['fig_size'])
    ax.set_xlabel('In-'+incoming_var.capitalize(), fontsize=fontsize_label)
    ax.set_ylabel('Out-'+incoming_var.capitalize(), fontsize=fontsize_label)
    artist = ax.hexbin(incoming, outgoing, gridsize=20,
                       cmap='gray_r', edgecolor='white')
    divider = make_axes_locatable(ax)
    cmap_pos = divider.append_axes('right', size="5%", pad="5%")
    cax = fig.add_axes(cmap_pos)
    cbar = fig.colorbar(artist, cax=cax)
    ax.spines['right'].set(visible=False)
    ax.spines['top'].set(visible=False)
    ax.tick_params(top=False, right=False)
    #cbar.set_ticks([5, 10, 15])
    cbar.ax.set_title('Bin Counts', ha='left', x=0)
    cbar.ax.tick_params(axis='y', color='white', left=True, right=True,
                        length=5, width=1.5)
    #cbar.outline.remove()
    #cbar.outline.set_visible(False)
    save_in_out_hist_fig =f'{output_path}/in_out_hist_{incoming_var}_{kargs["fig_name"]}'
    plt.savefig(save_in_out_hist_fig, dpi=500)
    plt.close(fig)

    # in vs clustering coeff histogram
    fig, ax = plt.subplots(figsize=kargs['fig_size'])
    ax.set_xlabel('In-'+incoming_var.capitalize(), fontsize=fontsize_label)
    ax.set_ylabel('Clustering coeff.', fontsize=fontsize_label)
    artist = ax.hexbin(incoming, clustering_coeff, gridsize=20,
                       cmap='gray_r', edgecolor='white')
    divider = make_axes_locatable(ax)
    cmap_pos = divider.append_axes('right', size="5%", pad="5%")
    cax = fig.add_axes(cmap_pos)
    cbar = fig.colorbar(artist, cax=cax)
    ax.spines['right'].set(visible=False)
    ax.spines['top'].set(visible=False)
    ax.tick_params(top=False, right=False)
    #cbar.set_ticks([5, 10, 15])
    cbar.ax.set_title('Bin Counts', ha='left', x=0)
    cbar.ax.tick_params(axis='y', color='white', left=True, right=True,
                        length=5, width=1.5)
    #cbar.outline.remove()
    #cbar.outline.set_visible(False)
    save_in_cc_hist_fig =f'{output_path}/in_cc_hist_{incoming_var}_{kargs["fig_name"]}'
    plt.savefig(save_in_cc_hist_fig, dpi=500)
    plt.close(fig)

    # out vs cc histogram
    fig, ax = plt.subplots(figsize=kargs['fig_size'])
    ax.set_xlabel('Out-'+incoming_var.capitalize(), fontsize=fontsize_label)
    ax.set_ylabel('Clustering coeff.', fontsize=fontsize_label)
    artist = ax.hexbin(outgoing, clustering_coeff, gridsize=20,
                       cmap='gray_r', edgecolor='white')
    divider = make_axes_locatable(ax)
    cmap_pos = divider.append_axes('right', size="5%", pad="5%")
    cax = fig.add_axes(cmap_pos)
    cbar = fig.colorbar(artist, cax=cax)
    ax.spines['right'].set(visible=False)
    ax.spines['top'].set(visible=False)
    ax.tick_params(top=False, right=False)
    #cbar.set_ticks([5, 10, 15])
    cbar.ax.set_title('Bin Counts', ha='left', x=0)
    cbar.ax.tick_params(axis='y', color='white', left=True, right=True,
                        length=5, width=1.5)
    #cbar.outline.remove()
    #cbar.outline.set_visible(False)
    save_out_cc_hist_fig =f'{output_path}/out_cc_hist_{incoming_var}_{kargs["fig_name"]}'
    plt.savefig(save_out_cc_hist_fig, dpi=500)
    plt.close(fig)

    # both vs cc histogram
    fig, ax = plt.subplots(figsize=kargs['fig_size'])
    ax.set_xlabel(incoming_var.capitalize(), fontsize=fontsize_label)
    ax.set_ylabel('Clustering coeff.', fontsize=fontsize_label)
    artist = ax.hexbin(both, clustering_coeff, gridsize=20,
                       cmap='gray_r', edgecolor='white')
    divider = make_axes_locatable(ax)
    cmap_pos = divider.append_axes('right', size="5%", pad="5%")
    cax = fig.add_axes(cmap_pos)
    cbar = fig.colorbar(artist, cax=cax)
    ax.spines['right'].set(visible=False)
    ax.spines['top'].set(visible=False)
    ax.tick_params(top=False, right=False)
    #cbar.set_ticks([5, 10, 15])
    cbar.ax.set_title('Bin Counts', ha='left', x=0)
    cbar.ax.tick_params(axis='y', color='white', left=True, right=True,
                        length=5, width=1.5)
    #cbar.outline.remove()
    #cbar.outline.set_visible(False)
    save_both_cc_hist_fig =f'{output_path}/both_cc_hist_{incoming_var}_{kargs["fig_name"]}'
    plt.savefig(save_both_cc_hist_fig, dpi=500)
    plt.close(fig)

    # in vs out vs clusterign coeff
    fig, ax = plt.subplots(figsize=kargs['fig_size'])
    #fig.suptitle(f'Clustering vs in-{incoming_var}',
    #             fontsize=fontsize_title)
    ax.set_ylabel('In-'+incoming_var.capitalize(), fontsize=fontsize_label)
    ax.set_xlabel('Out-'+incoming_var.capitalize(), fontsize=fontsize_label)
    points = ax.scatter(incoming, outgoing, c=clustering_coeff,
                        cmap=kargs['cmap'],
                        edgecolors=kargs['edgecolors'],
                        lw=kargs['lw'],
                        s=kargs['size'])
    divider = make_axes_locatable(ax)
    cmap_pos = divider.append_axes('right', size="5%", pad="5%")
    cax = fig.add_axes(cmap_pos)
    cbar = fig.colorbar(points, cax=cax)
    cbar.ax.set_title('Clustering Coeff.', ha='left', x=0)
    # save image
    save_cc_vs_in_fig =f'{output_path}/cc_vs_{incoming_var}_{kargs["fig_name"]}'
    plt.savefig(save_cc_vs_in_fig, dpi=500)
    plt.close(fig)

