"""
test_utils.py
    script for testing utilities
"""
import pytest

from src.utils import bionet_tools


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
