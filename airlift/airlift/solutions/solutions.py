import random
import time
import timeit

from gym.utils import seeding


class Solution:
    """Bare-bones class for creating a solution. While we do encourage you to use this as a starting point to create
    your own solution, it is not a requirement"""
    def __init__(self):
        self._np_random = None
        self.agents = None

    @property
    def name(self):
        """
        A property that contains the name of the solution.
        """
        return type(self).__name__

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        """
        Resets the solution with a seed. initializes  the state space, observation space and action space.

        :parameter obs: A dictionary containing the observation
        :parameter observation_spaces: A dictionary containing all the observation spaces
        :parameter action_spaces: A dictionary containing all the action spaces.
        :parameter seed: Environment seed.

        """
        # If a new seed is passed in, re-seed the solution.
        # If solution has not been reset yet, also do the seeding regardless.
        if seed is not None or self._np_random is None:
            self._np_random, seed = seeding.np_random(seed)

        self.agents = obs.keys()
        self.state_space = list(obs.values())[0]["globalstate"] # Assume state space for each agent is the same
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def policies(self, obs, dones, infos=None):
        """
        The main policy for returning actions for each agent is contained here.

        :parameter obs: A dictionary containing an observation
        :parameter dones: A dictionary containing the done values for each agent
        :return: Actions for each agent.
        """

        raise NotImplementedError

    @staticmethod
    def get_state(obs):
        """Gets the state for each agent. Assumes that the state space for each agent is the same"""
        return list(obs.values())[0]["globalstate"]  # Assume state space for each agent is the same


def doepisode(env, solution, render=False, env_seed=None, solution_seed=None, render_sleep_time=0.1, capture_metrics=False, render_mode="human"):
    """
    Runs a single episode.

    :parameter env: AirLiftEnv - An initialized Airlift Environment
    :parameter solution: Solution - the solution that is being utilized
    :parameter render: Render options, (video, window, none...)
    :parameter env_seed: int, environment seed,
    :parameter solution_seed: int, solution seed
    :parameter render_sleep_time: float, sleep timer
    :return: `env.env_info`: a NamedTuple that contains all the environment initialization parameters
        `env.metrics`: a NamedTuple that contains all the environment metrics collected for the solution.

    """

    # Run a single episode here
    step = 0
    _done = False
    infos = None
    obs = env.reset(seed=env_seed)
    if capture_metrics:
        step_metrics = [env.metrics]

    solution.reset(obs, env.observation_spaces, env.action_spaces, solution_seed)
    episode_starting_time = timeit.default_timer()
    starting_time = timeit.default_timer()
    total_solution_time = timeit.default_timer() - starting_time
    while not _done:
        # Compute Action
        starting_time = timeit.default_timer()
        actions = solution.policies(env.observe(), env.dones, infos=infos)
        total_solution_time += timeit.default_timer() - starting_time
        obs, rewards, dones, infos = env.step(actions)  # If there is no observation, just return 0
        if capture_metrics:
            step_metrics.append(env.metrics)

        _done = all(dones.values())
        step += 1
        if render:
            env.render(render_mode)
            if render_mode != "video":
                time.sleep(render_sleep_time)
    # print('is done')
    time_taken = timeit.default_timer() - episode_starting_time

    return_val = (env.env_info, env.metrics, time_taken, total_solution_time)
    if capture_metrics:
        return return_val + (step_metrics,)
    else:
        return return_val