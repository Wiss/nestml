#import os
import matplotlib.pyplot as plt

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
    # This should be here, it's just a test
    w = {}
    valid_pair = []
    for source, target, time, weight in zip(weights_events['ex_ex']['senders'],
                                         weights_events['ex_ex']['targets'],
                                         weights_events['ex_ex']['times'],
                                         weights_events['ex_ex']['weights']):

        valid_pair.append([source, target])
        print('source')
        print(source)
        print('target')
        print(target)
        print('time')
        print(time)
        print('weight')
        print(weight)

        #if source in w.keys():
       # if not bool(w):
       #     #if target in w[source].keys():
       #     if bool(w[source]):
       #         print(f'{target} key in w[source]')
       #         w[source][target]['times'].append(time)
       #         w[source][target]['weights'].append(weight)
       #     else:
       #         print(f'no {target} key in w[source]')
       #         w[source][target] = {'times': [], 'weights': []}
       # else:
       #     print(f'no {source} key in w')
       #     w[source] = dict

        w.setdefault(str(source), {})
        w[str(source)].setdefault(str(target), {})
        w[str(source)][str(target)].setdefault('times', []).append(time)
        w[str(source)][str(target)].setdefault('weights', []).append(weight)

    print(w)
    plt.figure()
    sources_set = weights_events['ex_ex']['senders']
    target_set = weights_events['ex_ex']['targets']
    #valid_pair_set = set(valid_pair)
    valid_pair_set = [i for n, i in enumerate(valid_pair) if i not in valid_pair[:n]]
    for s, t in valid_pair_set:
        plt.plot(w[str(s)][str(t)]['times'], w[str(s)][str(t)]['weights'], '.', label=f'{s}->{t}')
        plt.plot(w[str(s)][str(t)]['times'], w[str(s)][str(t)]['weights'])
    #for s in sources_set:
     #   for t in target_set:
      #      try:
       #         plt.plot(w['3'][t]['times'], w['3'][t]['weights'], label=f'{s}->{t}')
       #     except:
        #        pass
    #plt.plot(w['3']['2']['times'], w['3']['2']['weights'], label=f'{3}->{2}')
    #plt.plot(w['3']['2']['times'], w['3']['2']['weights'], '.', label=f'{3}->{2}')
    #plt.plot(w['3']['3']['times'], w['3']['3']['weights'], label=f'{3}->{3}')
    #plt.plot(w['3']['3']['times'], w['3']['3']['weights'], '.', label=f'{3}->{3}')
    #plt.plot(w['3']['4']['times'], w['3']['4']['weights'], label=f'{3}->{4}')
    #plt.plot(w['3']['4']['times'], w['3']['4']['weights'], '.', label=f'{3}->{4}')
    plt.xlabel('times (ms)')
    plt.ylabel('weights')
    plt.legend()
    plt.show()
    pass

def create_spikes_figs(spikes_events: dict, fig_name: str, output_path: str,
                     **kargs):
    """
    Raster plots

    Parameters
    ----------
    spikes_dict:
        dictionary with all spike's information (events)
    """
    fig, ax = plt.subplots(2*len(spikes_events), figsize=fig_size, sharex=True)
    ax[-1].set_xlabel('time (ms)', fontsize=fontsize_label)
    n = 0
    for pop, events in spikes_events.items():
        if pop == 'ex':
            color = 'r'
            p = 'excitatory'
        elif pop == 'in':
            color = 'b'
            p = 'inhibitory'
        #fig, ax = plt.subplots(, figsize=fig_size)
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
    fig, ax = plt.subplots(2, figsize=fig_size, sharex=True)
    ax[0].set_title('Spikes', fontsize=fontsize_title)
    ax[1].set_title('Synchronization', fontsize=fontsize_title)
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
        ax[0].plot(times, senders, '.', c=color, label=pop)
        ax[0].set_ylabel('Neuron ID', fontsize=fontsize_label)
        ax[0].legend(fontsize=fontsize_legend)
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
