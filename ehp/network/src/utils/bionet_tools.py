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


def set_seed(seed):
    """
    set seed for numpy and nest
    """
    logger.info("Setting seed %i for nest and numpy", seed)
    nest.rng_seed = seed
    np.random.seed(seed)



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
        nest.Install(module_name)
        nest.Create(neuron_model)
    except:
        nest.ResetKernel()
        generate_nest_target(input_path="models/neurons/" + neuron_model + ".nestml",
                             target_path="/tmp/nestml-component",
                             module_name=module_name,
                             logging_level="INFO",
                             codegen_opts={"nest_path":
                                           NEST_SIMULATOR_INSTALL_LOCATION})
        nest.Install(module_name)


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
    # this shouldn't be here. Organice!
    nest.PlotLayer(pop, nodesize=80)
    plt.show()
    return pop

def fix_syn_spec(syn_spec: dict):
    """
    given config values for connections, create synaptic specifications
    following nest syntaxys

    Parameters
    ----------
    syn_spec:
        synaptic specifications from config file
    """
    syn_spec_fixed = {}
    syn_spec_fixed["synapse_model"] = syn_spec["synapse_model"]
    # set alpha values
    if syn_spec["weight"]["dist"]:
        if syn_spec["weight"]["dist"] == "exponential":
            syn_spec_fixed["weight"] = nest.random.exponential(
                                            beta=syn_spec["weight"]["beta"])
        else:
            raise KeyError

    # set delay values
    if syn_spec["delay"]["dist"]:
        if syn_spec["delay"]["dist"] == "uniform":
            syn_spec_fixed["delay"] = nest.random.uniform(
                                                    min=syn_spec["delay"]["min"],
                                                    max=syn_spec["delay"]["max"])
        else:
            raise KeyError

    # set alpha values
    if syn_spec["alpha"]["dist"]:
        # do not include this param if we have a static synapse
        if syn_spec["synapse_model"] == "static_synapse":
            logger.error("This conection doesn't allow alpha params")
            raise TypeError("This conection doesn't allow alpha params")
        if syn_spec["alpha"]["dist"] == "uniform":
            syn_spec_fixed["alpha"] = nest.random.uniform(
                                                min=syn_spec["alpha"]["min"],
                                                max=syn_spec["alpha"]["max"])
        else:
            raise KeyError
    return syn_spec_fixed

def include_params(syn_spec: dict, params: dict):
    """
    include syn_spec['params'] (config file) into syn_spec

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

def get_connections(pop_pre, pop_post):
    """
    read all weights values from pop_pre to pop_post

    pop could also be a subpopulation

    Parameters
    ----------

    pop_pre:
        presynaptic (sub)population
    pop_post:
        postsynaptic (sub)population
    """
    syn_coll = nest.GetConnections(pop_pre, pop_post)
    #logger.debug(syn_coll.weight)
    return syn_coll

def connect_pops(pop_pre, pop_post, conn_spec: dict, syn_spec: dict,
               label: str, record: bool = False):
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
    record:
        do we want to record weights?
    """
    # fix syn_dict
    syn_spec_fixed = fix_syn_spec(syn_spec)

    # include syn_spec["params"] from config file if we have plasticity
    if syn_spec['synapse_model'] != 'static_synapse':
        # FIX
        include_params(syn_spec_fixed, syn_spec['params'])
    # create recorder if we are recording
    if record:
        pass
        # TODO
        #wr = nest.Create('weight_recorder', label=label)
        #nest.CopyModel(syn_spec['synapse_model'],
        #               f'{label}_rec')
        #syn_spec_fixed['weight_recorder'] = wr

    conn = nest.Connect(pop_pre, pop_post, conn_spec=conn_spec,
                        syn_spec=syn_spec_fixed)

    syn_coll = get_connections(pop_pre, pop_post)
    logger.debug("connections for %s generated", label)
    logger.debug(nest.GetConnections(pop_pre, pop_post))
    return conn, syn_coll

def simulate(simtime: float):
    """
    simulate the network

    Parameters
    ----------
    simitme:
        simulation time in ms
    TODO:
    here we can include some protocolos for reading data and
    return it to the experiment
    """
