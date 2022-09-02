import matplotlib.pyplot as plt
import nest
import numpy as np

import os
import sys

from pynestml.frontend.pynestml_frontend import generate_nest_target
NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")


def init_population(neuron_model: str, n_neurons: int, params: dict,
                  pos_bounds: list, dim: int):
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
    pop_pos = nest.spatial.free(
                    nest.random.uniform(min=min(pos_bounds),
                                        max=max(pos_bounds)),
                    num_dimensions=dim
                    )
    pop = nest.Create(neuron_model, n=n_neurons,
                      positions=pop_pos)
    for param in params:
        pop.param = nest.random.normal(mean=param['mean'], std=param['std'])
    return pop

def connect_pops(pop_pre,
               pop_post,
               weights: list):
    """
    initialize weights between two populations

    Parameters
    ----------
    pop_pre: nest population
        presynaptic population
    pop_post: nest population
        postsynaptic population
    weights:
        weights between the two populations
    """
    nest.Connect(pop_pre, pop_post)


def connect_noise():
    pass


def measure():
    pass


def init_network(neurons: dict, connections: dict, network_layout: dict,
               external_source: dict):
    """
    Initialize network

    Parameters
    ----------
    neurons:
        neuron parameters
    connections:
        connection parameters
    network_layout:
        network layout parameters
    external_source:
        external source parameters
    """
    # init pops
    pop_ex = init_population()
    pop_in = init_population()
    # connect both populations
    connect_pops(pop_ex, pop_in)
    # connect external sources
    connect_noise()
    # connect measurement devices
    measure()

def run_network(simtime: float, seed: int, record: bool, record_rate: float):
    """
    run network

    Parameters
    ----------
    simtime:
        simulation time
    seed:
        random seed
    record:
        do you want to record results?
    record_rate:
        recording rate

    Returns
    -------
    sim_results: dict

    """
    sim_results = {}
    nest.Simulate(simtime)
    return sim_results
