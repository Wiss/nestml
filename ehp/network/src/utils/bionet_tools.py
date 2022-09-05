"""
bionet_tool.py
    tools for constructing and manage the biophysical network
"""
import nest
import numpy as np

from src.logging.logging import logger

from pynestml.frontend.pynestml_frontend import generate_nest_target
NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")

logger = logger.getChild(__name__)

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
    # TODO: this shoukld be fixed takinmg into account which neurons and
    # synapses are being used
    if neuron_model == "edlif_psc_exp_percent":
        module_name = "edlif_psc_exp" + "_module"
        try_install_module(module_name, neuron_model)
    elif neuron_model == "edlif_psc_alpha_percent":
        module_name = "edlif_psc_alpha" + "_module"
        try_install_module(module_name, neuron_model)
    nest.ResetKernel()

    # define neuron positions
    if position_dist == "uniform":
        pop_pos = nest.spatial.free(
                        nest.random.uniform(min=min(pos_bounds),
                                            max=max(pos_bounds)),
                        num_dimensions=dim
                        )
        pop = nest.Create(neuron_model, n=n_neurons,
                          positions=pop_pos)
        logger.debug("uniform distribution created for %s positions",
                     neuron_model)

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
        logger.debug("seting general param %s", param)
        logger.debug("with mean: %s and std: %s", param_v['mean'],
                     param_v['std'])
        pop.set({param: [param_v['mean'] +
                         param_v['std']*np.random.rand() for x in range(len(pop))]})
        logger.debug(pop.get(param))
    return pop
