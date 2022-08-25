import matplotlib.pyplot as plt
import nest
import numpy as np

import os
import sys

from pynestml.frontend.pynestml_frontend import generate_nest_target
NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")
