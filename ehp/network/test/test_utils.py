"""
test_utils.py
    script for testing utilities
"""
import pytest

from src.utils import bionet_tools


@pytest.fixture
def population():
    pos_dist = "uniform"
    n_model = "iaf_psc_alpha"
    n_neurons = 10
    params = {"general_params":
              {"tau_m": {
                  "mean": 20,
                  "std": 0.1
              },
               "tau_syn_ex": {
                   "mean": 30,
                   "std": 0
               },
               "C_m": {
                   "mean": 220,
                   "std": 10
               },
               "t_ref": {
                   "mean": 1,
                   "std": 0.01
               }},
              "energy_params":
              {"ATP": {
                  "mean": 97,
                  "std": 1
              }
               }}
    pos_bounds = [-1, 1]
    dim = 2
    pop1 = bionet_tools.init_population(position_dist=pos_dist,
                                       neuron_model=n_model,
                                       n_neurons=n_neurons,
                                       params=params,
                                       pos_bounds=pos_bounds,
                                       dim=dim)
    pop2 = bionet_tools.init_population(position_dist=pos_dist,
                                       neuron_model=n_model,
                                       n_neurons=n_neurons,
                                       params=params,
                                       pos_bounds=pos_bounds,
                                       dim=dim)
    return pop1, pop2


@pytest.mark.parametrize(
    "pos_dist, n_model, n_neurons, params, pos_bounds, dim, expected",
    [("uniform", "iaf_psc_alpha", 5, {"general_params":
                                      {"tau_m": {
                                                "mean": 20,
                                                "std": 0.1
                                                },
                                      "tau_syn_ex": {
                                                "mean": 30,
                                                "std": 0
                                                },
                                      "C_m": {
                                                "mean": 220,
                                                "std": 10
                                                },
                                      "t_ref": {
                                                "mean": 1,
                                                "std": 0.01
                                                }},
                                      "energy_params":
                                      {"ATP": {
                                                "mean": 97,
                                                "std": 1
                                              }
                                      }},
      [-1, 1], 2, 5),
     ("uniform", "edlif_psc_alpha_percent", 5, {"general_params":
                                      {"tau_m": {
                                                "mean": 20,
                                                "std": 0.1
                                                },
                                      "tau_syn_ex": {
                                                "mean": 30,
                                                "std": 0
                                                },
                                      "C_m": {
                                                "mean": 220,
                                                "std": 10
                                                },
                                      "t_ref": {
                                                "mean": 1,
                                                "std": 0.01
                                                }},
                                      "energy_params":
                                      {"tau_ap": {
                                                "mean": 30.,
                                                "std": 1
                                              }
                                      }},
      [-1, 1], 2, 5)
     ],)
def test__init_population__succeed(pos_dist, n_model, n_neurons, params,
                                pos_bounds, dim, expected):
    """
    test population initialization
    """
    pop = bionet_tools.init_population(position_dist=pos_dist,
                                       neuron_model=n_model,
                                       n_neurons=n_neurons,
                                       params=params,
                                       pos_bounds=pos_bounds,
                                       dim=dim)
    assert len(pop) == expected


@pytest.mark.parametrize(
    "conn_spec, syn_spec, expected",
    [({"allow_autapses": True,
       "allow_multapses": False,
       "rule": "pairwise_bernoulli",
       "p": 0.2},
      {"synapse_model": "stdp_synapse",
       "weights_dist": "exponential",
       "delays_dist": "uniform",
       "alphas_dist": "uniform"},
      0)
     ],)
def test__init_network__succeed(population, conn_spec, syn_spec, expected):
    """
    test connection initialization
    """
    pop1, pop2 = population[0], population[1]
    bionet_tools.connect_pops(pop_pre=pop1,
                              pop_post=pop2,
                              conn_spec=conn_spec,
                              syn_spec=syn_spec
                              )
    #assert len(pop) == expected
