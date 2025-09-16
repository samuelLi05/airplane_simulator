from airlift.solutions.solutions import doepisode
from gym.utils import seeding

from airlift.solutions.baselines import RandomAgent, ShortestPath
from airlift.envs.events.event_interval_generator import  EventIntervalGenerator
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator
from airlift.utils.history_wrapper import HistoryWrapper, obs_equal, assert_histories_equal
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.plane_types import PlaneType
import pytest
import networkx





def test_deterministic_lockstep_random(render):
    run_lockstep(gen_env(), gen_env(), RandomAgent(), RandomAgent())

def test_deterministic_lockstep_shortest(render):
    run_lockstep(gen_env(), gen_env(), ShortestPath(), ShortestPath())

def test_deterministic_one_at_a_time_random(render):
    run_one_at_a_time(gen_env(), gen_env(), RandomAgent(), RandomAgent())

def test_deterministic_one_at_a_time_shortest(render):
    run_one_at_a_time(gen_env(), gen_env(), ShortestPath(), ShortestPath())

def test_deterministic_parallel_random(render):
    run_in_parallel(gen_env(), gen_env(), ShortestPath(), ShortestPath())

def test_deterministic_parallel_shortest(render):
    run_in_parallel(gen_env(), gen_env(), ShortestPath(), ShortestPath())


# TODO: Run evaluator to ensure it gives determinstic results



# Helper methods
def gen_env():
    single_plane_type = [PlaneType(id=0, max_range=2, speed=0.4, max_weight=10)]
    return AirliftEnv(
        world_generator=AirliftWorldGenerator(
            plane_types=single_plane_type,
            airport_generator=RandomAirportGenerator(
                max_airports=10,
                processing_time=10,
                working_capacity=2,
                make_drop_off_area=False,
                make_pick_up_area=False,
                num_drop_off_airports=1,
                num_pick_up_airports=1
            ),
            route_generator=RouteByDistanceGenerator(
                malfunction_generator=EventIntervalGenerator(
                    min_duration=10,
                    max_duration=100), poisson_lambda=.05,
                route_ratio=2
            ),
            cargo_generator=DynamicCargoGenerator(
                cargo_creation_rate=1 / 100,
                max_cargo_to_create=5,
                num_initial_tasks=5,
                soft_deadline_multiplier=45,
                hard_deadline_multiplier=50
            ),
            airplane_generator=AirplaneGenerator(20),
            max_cycles=100
        )
    )


def run_lockstep(env1, env2, solution1, solution2):
    # Env Reset
    obs1 = env1.reset(seed=54554455555)
    obs2 = env2.reset(seed=54554455555)
    assert obs_equal(obs1, obs2)

    # Solutions Reset
    solution1.reset(env1.observe(), seed=35555)
    solution2.reset(env2.observe(), seed=35555)

    _done = False
    while not _done:
        actions1 = solution1.policies(env1.observe(), env1.dones)
        actions2 = solution2.policies(env2.observe(), env2.dones)

        obs1, rewards1, dones1, infos1 = env1.step(actions1)
        obs2, rewards2, dones2, infos2 = env2.step(actions2)

        assert actions1 == actions2
        assert rewards1 == rewards2
        assert dones1 == dones2
        assert obs_equal(obs1, obs2)
        assert infos1 == infos2

        _done = all(dones1) and all(dones2)


def run_with_history(env, solution):
    env = HistoryWrapper(env)
    doepisode(env,
              solution=solution,
              render=False,
              env_seed=54554455555,
              solution_seed=35555)

    return env.history

def run_one_at_a_time(env1, env2, solution1, solution2):
    history1 = run_with_history(env1, solution1)
    history2 = run_with_history(env2, solution2)
    assert_histories_equal(history1, history2)


def run_in_parallel(env1, env2, solution1, solution2):
    from joblib import Parallel, delayed
    histories = Parallel(n_jobs=2)(
                    [delayed(run_with_history)(env1, solution1),
                     delayed(run_with_history)(env2, solution2)])

    assert_histories_equal(histories[0], histories[1])



