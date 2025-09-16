import gym
import numpy as np

from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.tests.util import generate_environment
from gym import logger

logger.set_level(logger.WARN)

def test_global_state_flatten(render):
    env = generate_environment(num_of_airports=3, num_of_agents=1, cargo_generator=StaticCargoGenerator(1))
    obs = env.reset(seed=546435)

    flat_dim = gym.spaces.flatdim(env.state_space())
    assert isinstance(flat_dim, int)

    flat_state = gym.spaces.flatten(env.state_space(), env.state())
    assert isinstance(flat_state, (np.ndarray, np.generic))

    # Space flattening/unflattening is not implemented
    # flat_to_box = gym.spaces.flatten_space(env.state_space())
    # assert isinstance(flat_to_box, Box)

    # Unflattening is not implemented (yet, but maybe in the future)
    # unflat_state = gym.spaces.unflatten(env.state_space(), flat_state)
    # assert isinstance(unflat_state, Space)


def test_agent_obs_flatten(render):
    env = generate_environment(num_of_airports=3, num_of_agents=1, cargo_generator=StaticCargoGenerator(1)) # 1 task
    obs = env.reset(seed=546435)

    for a in env.agents:
        flat_dim = gym.spaces.flatdim(env.observation_space(a))
        assert isinstance(flat_dim, int)

        # Space flattening/unflattening is not implemented
        # flat_to_box = gym.spaces.flatten_space(env.observation_space(a))
        # assert isinstance(flat_to_box, Box)

        flat_obs = gym.spaces.flatten(env.observation_space(a), env.observe(a))
        assert isinstance(flat_obs, (np.ndarray, np.generic))

        # Unflattening is not implemented (yet, but maybe in the future)
        # unflat_obs = gym.spaces.unflatten(env.observation_space(a), flat_obs)
        # assert isinstance(unflat_obs, Space)
