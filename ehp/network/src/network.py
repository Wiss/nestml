#import matplotlib.pyplot as plt
#import nest
#import numpy as np
#import os
#import sys

from src.logging.logging import logger
from src.utils.bionet_tools import (init_population, connect_pops,
                                    set_seed, simulate)
#from pynestml.frontend.pynestml_frontend import generate_nest_target
#NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")

logger = logger.getChild(__name__)



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
    position_dist = network_layout["positions"]["dist"]
    n_neurons = network_layout["n_neurons"]
    ex_in_ratio = network_layout["ex_in_ratio"]
    n_neurons_ex = int(n_neurons * ex_in_ratio)
    n_neurons_in = n_neurons - n_neurons_ex
    logger.info("Initializing network with %i neurons", n_neurons)
    logger.info("%i excitatory and %i inhibitory neurons", n_neurons_ex,
                n_neurons_in)
    pop_ex = init_population(position_dist=position_dist,
                             neuron_model=neurons["ex"]["model"],
                             n_neurons=n_neurons_ex,
                             params=neurons["ex"]["params"],
                             pos_bounds=network_layout["positions"]["pos_bounds"],
                             dim=network_layout["positions"]["dim"]
                             )
    pop_in = init_population(position_dist=position_dist,
                             neuron_model=neurons["in"]["model"],
                             n_neurons=n_neurons_in,
                             params=neurons["in"]["params"],
                             pos_bounds=network_layout["positions"]["pos_bounds"],
                             dim=network_layout["positions"]["dim"]
                             )
    # Create connections
    for conn, conn_v in connections.items():
        print(connections)
        print(conn)
        print(conn_v)
        if conn in ['ex_ex', 'ex_in', 'in_ex', 'in_in']:
            logger.info("Setting up connection %s -> %s",
                        conn.split("_")[0], conn.split("_")[1])
            if conn.split("_")[0] == conn.split("_")[1] == "ex":
                connect_pops(pop_ex, pop_ex, conn_v["conn_spec"],
                             conn_v["syn_spec"], label=conn)
                pass
            elif conn.split("_")[0] == conn.split("_")[1] == "in":
                connect_pops(pop_in, pop_in, conn_v["conn_spec"],
                             conn_v["syn_spec"], label=conn)
                pass
            elif conn.split("_")[0] == "ex" and conn.split("_")[1] == "in":
                connect_pops(pop_ex, pop_in, conn_v["conn_spec"],
                             conn_v["syn_spec"], label=conn)
                pass
            elif conn.split("_")[0] == "in" and conn.split("_")[1] == "ex":
                connect_pops(pop_in, pop_ex, conn_v["conn_spec"],
                             conn_v["syn_spec"], label=conn)
                pass

    # connect external sources
    connect_noise()
    # connect measurement devices
    measure()
    return pop_ex, pop_in

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
    # simulation results
    sim_results = {}
    # set seed for simulation
    set_seed(seed)
    simulate(simtime)
    return sim_results
