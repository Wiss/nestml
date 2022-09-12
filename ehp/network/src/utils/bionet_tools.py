"""
bionet_tool.py
    tools for constructing and manage the biophysical network
"""
import nest
import matplotlib.pyplot as plt
import numpy as np

from src.logging.logging import logger

from pynestml.frontend.pynestml_frontend import generate_nest_target
NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")

logger = logger.getChild(__name__)


def set_seed(seed: int):
    """
    set seed for numpy and nest

    Parameters
    ----------
    seed
    """
    logger.info("Setting seed %i for nest and numpy", seed)
    nest.rng_seed = seed
    np.random.seed(seed)


def reset_kernel():
     nest.ResetKernel


def install_needed_modules():
    """
    Install needed nestml modules

    Parameters
    ----------

    """
    pass

def try_install_module(module_name, neuron_model):
    """
    try loading neuron model
    """
    try:
        nest.Create(neuron_model)
        logger.debug('model %s already loaded', neuron_model)
    except:
        nest.Install(module_name)
        nest.Create(neuron_model)
        logger.debug('model %s installed and loaded', neuron_model)
    #else:
        #nest.ResetKernel()
        #generate_nest_target(input_path="models/neurons/" + neuron_model + ".nestml",
        #                     target_path="/tmp/nestml-component",
        #                     module_name=module_name,
        #                     logging_level="INFO",
        #                     codegen_opts={"nest_path":
        #                                   NEST_SIMULATOR_INSTALL_LOCATION})
        #nest.Install(module_name)


def init_population(position_dist: str, neuron_model: str, n_neurons: int,
                  params: dict, pos_bounds: list, dim: int):
    """
    initialize neuron population

    Parameters
    ----------
    neuron_model:
        neuron model's name
    n_neurons:
        how many neurons in that population
    params:
        dictionary with parameters for the population
    pos_bounds:
        list with min and max random position values
    dim:
        spatial dimension
    """
    # check if model is edlif
    # TODO: this should be fixed taking into account which neurons and
    # synapses are being used
    if neuron_model == "edlif_psc_exp_percent":
        module_name = "edlif_psc_exp" + "_module"
        try_install_module(module_name, neuron_model)
    elif neuron_model == "edlif_psc_alpha_percent":
        module_name = "edlif_psc_alpha" + "_module"
        try_install_module(module_name, neuron_model)

    # define neuron positions
    if position_dist == "uniform":
        pop_pos = nest.spatial.free(
                        nest.random.uniform(min=min(pos_bounds),
                                            max=max(pos_bounds)),
                        num_dimensions=dim
                        )
        pop = nest.Create(neuron_model, n=n_neurons,
                          positions=pop_pos)
        logger.debug("uniform distribution with %i neurons created, for %s"
                        " positions", n_neurons, neuron_model)

    if "edlif" in neuron_model.split("_"):  # check if the model es ED
        # Energy params only for energy-dependent neurons
        for param, param_v in params["energy_params"].items():
            logger.debug("seting energy param %s", param)
            logger.debug("with mean: %s and std: %s", param_v['mean'],
                         param_v['std'])
            pop.set({param: [param_v['mean'] +
                             param_v['std']*np.random.rand() for x in range(len(pop))]})
            logger.debug(pop.get(param))

    # General params for all neuron types
    for param, param_v in params["general_params"].items():
        logger.debug("setting general param %s", param)
        logger.debug("with mean: %s and std: %s", param_v['mean'],
                     param_v['std'])
        pop.set({param: [param_v['mean'] +
                         param_v['std']*np.random.rand() for x in range(len(pop))]})
        logger.debug(pop.get(param))
    return pop

def fix_syn_spec(syn_spec: dict, label: str):
    """
    given config values for connections, create synaptic specifications
    following nest syntaxys

    Parameters
    ----------
    syn_spec:
        synaptic specifications from config file
    label:
        connection label
    """
    syn_spec_fixed = {}
    syn_spec_fixed["synapse_model"] = syn_spec["synapse_model"]
    # set weight, delay and alpha values
    for key in syn_spec.keys():
        if key in ['weight', 'delay', 'alpha', 'w']:
            logger.info('connection param "%s" with specifications: %s',
                        key, syn_spec[key])
            if syn_spec[key]['dist']:
                if syn_spec[key]['dist'] == 'exponential':
                    syn_spec_fixed[key] = nest.random.exponential(
                                                beta=syn_spec[key]["beta"])
                    # set megative weights if pop_pre is inhibitory
                    if label.split("_")[0] == "in" and key in ['weight', 'w']:
                        syn_spec_fixed[key] *= -1

                elif syn_spec[key]['dist'] == 'uniform':
                    syn_spec_fixed[key] = nest.random.uniform(
                                                    min=syn_spec[key]["min"],
                                                    max=syn_spec[key]["max"])
            if (key == 'delay' and 'edlif' in
                syn_spec['synapse_model'].split('_')):
                # It's neccesary to do this, because ed_stpd cannot
                # have a variable called delay
                syn_spec_fixed['d'] = v
    return syn_spec_fixed

def include_params(syn_spec: dict, params: dict):
    """
    include syn_spec['params'] (from config file) into syn_spec

    Parameters
    ----------
    syn_spec:
        synapses specifications dict
    params:
        extra params from config file
    """
    for k, v in params.items():
        if v is not None:
            logger.debug("including param %s = %f in syn_spec dict", k, v)
            syn_spec[k] = v
    return syn_spec

def get_connections(pop_pre: nest.NodeCollection, pop_post: nest.NodeCollection,
                  synapse_model: str):
    """
    read all weights values from pop_pre to pop_post

    pop could also be a subpopulation

    Parameters
    ----------

    pop_pre:
        presynaptic (sub)population
    pop_post:
        postsynaptic (sub)population
    synapse_model:
        synapse model name
    """
    #if "edlif" in synapse_model.split('_'):
    #    param_list = ['source', 'target', 'synapse model', 'w', 'delay']
    #else:
    #    param_list = ['source', 'target', 'synapse model', 'weight', 'delay']
    syn_coll = nest.GetConnections(source=pop_pre,
                                   target=pop_post,
                                   synapse_model=synapse_model)
    return syn_coll

def update_syn_w_wr(syn: nest.NodeCollection, syn_spec: dict, label: str):
    """
    uodate synaptic collection with weight recorder

    Parameters
    ----------
    syn: NodeCollection
        synaptic collection
    syn_spec:
        synaptic specifications
    label:
        connection label
    """
    for k, v in syn_spec.items():
        if k in ['weight', 'w', 'delay', 'alpha'] and v:
            logger.info('connection param (wr) "%s" with specifications: %s',
                        k, syn_spec[k])
            for s in syn:
                if syn_spec[k]['dist'] is not None:
                    if syn_spec[k]['dist'] == 'exponential':
                        v = np.random.exponential(syn_spec[k]['beta'])
                        # negative weight if pop_pre is inh
                        if label.split("_")[0] == "in" and k in ['weight', 'w']:
                            v *= -1
                    elif syn_spec[k]['dist'] == 'uniform':
                        v = np.random.uniform(low=syn_spec[k]['min'],
                                            high=syn_spec[k]['max'])
                    else:
                        raise KeyError(f'"{k}" key not supported yet')
                    # update synapses
                    new_param_dict = {k: v}
                    nest.SetStatus(s, new_param_dict)
                    if (k == 'delay' and 'edlif' in
                        syn_spec['synapse_model'].split('_')):
                        print("delay")
                        print(v)
                        # It's neccesary to do this, because ed_stpd cannot
                        # have a variable called delay
                        nest.SetStatus(s, {'d': v})
                # if dist == None, continue
                else:
                    continue

        elif k == 'params':
            for kp, vp in syn_spec[k].items():
                if vp is not None:
                    #for s in syn:
                    new_param_dict = {kp: vp}
                    nest.SetStatus(syn, new_param_dict)
        elif k in ['synapse_model', 'record']:
           continue
        else:
            raise KeyError(f'"{k}" key not supported yet')

def connect_pops(pop_pre: nest.NodeCollection, pop_post: nest.NodeCollection,
               conn_spec: dict, syn_spec: dict, label: str,
               weight_rec_list: list):
    """
    initialize weights between two populations

    Parameters
    ----------
    pop_pre: nest population
        presynaptic population
    pop_post: nest population
        postsynaptic population
    conn_spec:
        connection specifications
    syn_spec:
        synaptic specifications
    params:
        extra params for the synapses
    label:
        connection label
    weight_rec_list:
        this list will contain every needed weight recorder

    Return
    ------
    conn: dict
        connection dictionary
    """
    # create recorder if we are recording
    if syn_spec['record']:
        # create weight recorder
        weight_rec_list.append(nest.Create('weight_recorder'))
        # copy synaptic_model to record weights
        nest.CopyModel(syn_spec['synapse_model'],
                       f"{label}_rec",
                       {'weight_recorder': weight_rec_list[-1]})
        nest.Connect(pop_pre, pop_post,
                     conn_spec=conn_spec,
                     syn_spec={'synapse_model': f'{label}_rec'})
        logger.info("new weight recorder for %s label created", label)
        # get syn object
        syn = get_connections(pop_pre=pop_pre,
                              pop_post=pop_post,
                              synapse_model=f'{label}_rec')
        # update synapses with param from config file
        # this is necessary when working with syn objects
        update_syn_w_wr(syn=syn,
                        syn_spec=syn_spec,
                        label=label)
        logger.info("parameters for weight recorder %s_rec updated", label)
        conn = get_connections(pop_pre=pop_pre,
                               pop_post=pop_post,
                               synapse_model=f'{label}_rec')
    else:
        weight_rec_list.append(None)
        # fix syn_dict
        syn_spec_fixed = fix_syn_spec(syn_spec, label)
        # include syn_spec["params"] from config file if we have plasticity
        if syn_spec['synapse_model'] != 'static_synapse':
            # FIX
            include_params(syn_spec_fixed, syn_spec['params'])
        nest.Connect(pop_pre,
                     pop_post,
                     conn_spec=conn_spec,
                     syn_spec=syn_spec_fixed)

        conn = get_connections(pop_pre=pop_pre,
                               pop_post=pop_post,
                               synapse_model=syn_spec['synapse_model'])
        logger.debug("connections for %s generated", label)
    logger.debug(conn)
    return conn, weight_rec_list[-1]

def simulate(simtime: float, record: dict, record_rate: int, pop_dict: dict,
           weight_rec_dict: dict):
    """
    simulate the network

    Parameters
    ----------
    simitme:
        simulation time in ms
    record:
        dictionary with information about what should be recorded
    record_rate:
        recording rate
    pop_dict:
        dictionary with populations
    weight_rec_dict:
        dictionary with weight recorders
    """
    # record spikes
    sr = {}
    if record['spikes']:
        for pop_k, pop_v in pop_dict.items():
            sr[pop_k] = nest.Create('spike_recorder')
            logger.info("connecting %s population to spike recorder", pop_k)
            nest.Connect(pop_v, sr[pop_k])
    else:
        sr = None

    # record multimeter
    mult = {}
    if record['multimeter']:
        print(record['multimeter'])
        for pop_k, pop_v in pop_dict.items():
            if (sum([var in nest.GetDefaults(pop_v[0].model)['recordables']
                     for var in record['multimeter']]) ==
                    len(record['multimeter'])):
                mult[pop_k] = nest.Create('multimeter',
                                        params={'interval': record_rate,
                                        'record_from': record['multimeter']})
                nest.Connect(mult[pop_k], pop_v)
                logger.info("reading %s from %s population",
                            record['multimeter'], pop_k)
                logger.info("connecting %s population to multimeter", pop_k)
            else:
                mult[pop_k] = None

    weights = weight_rec_dict

    logger.info("running simulation")
    nest.Simulate(simtime)
    return sr, mult, weights
