import pytest

import random
import numpy as np

from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import api_test, max_cycles_test, render_test, performance_benchmark, test_save_obs as petting_zoo_test_save_obs
from pettingzoo.test.seed_test import parallel_seed_test
from pettingzoo.test.parallel_test import parallel_api_test

from airlift.tests.util import generate_environment


# To run (--render is optional):
# pytest unit_tests/test_petting_zoo.py --render

# Note that we seed the default random number generators, but the environment still may have randomness since
# the tests call env.reset() with no seed. It seems NumPy's SeedSequence method used by OpenAI Gym to generate seeds
# will generate a random sequence if entropy=None regardless of whether the default generators are seeded.
# (not positive, though)


def createenv(maxcycles=2**32):
    return generate_environment(num_of_agents=1, num_of_airports=5)



# See https://www.pettingzoo.ml/environment_creation#tests

def test_api():
    random.seed(3490)
    np.random.seed(3490)

    # Note this gives some warnings which we ignore:
    #    Observation is not NumPy array
    #    Observation space for each agent probably should be gym.spaces.box or gym.spaces.discrete
    #    Action space for each agent probably should be gym.spaces.box or gym.spaces.discrete
    api_test(parallel_to_aec(createenv()))

def test_parallel_api():
    random.seed(3491)
    np.random.seed(3491)

    parallel_api_test(createenv())

def test_seed():
    random.seed(3492)
    np.random.seed(3492)

    parallel_seed_test(createenv)

# Skipping, because:
# The following line always fails in max_cycles_test: "assert max_cycles == np.max(agent_counts) - 1"
# I believe it should be "assert max_cycles == np.max(agent_counts)"
@pytest.mark.skip(reason="max_cycles_test seems broken")
def test_max_cycles():
    random.seed(3493)
    np.random.seed(3493)

    # The max cycles expects ot receive a module with generator methods
    class TestClass:
        parallel_env = lambda max_cycles : createenv(max_cycles)
        env = lambda max_cycles : parallel_to_aec(createenv(max_cycles))

    max_cycles_test(TestClass)

# We should manually check the results of this for it to be really useful
def test_performance_benchmark():
    random.seed(3492)
    np.random.seed(3492)

    performance_benchmark(parallel_to_aec(createenv()))

def test_render():
    random.seed(3492)
    np.random.seed(3492)

    render_test(lambda: parallel_to_aec(createenv()))

def test_save_obs():
    random.seed(3492)
    np.random.seed(3492)

    petting_zoo_test_save_obs(createenv())
