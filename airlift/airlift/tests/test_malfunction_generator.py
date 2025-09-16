from gym.utils import seeding

from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.events.event_interval_generator import EventIntervalGenerator, NoEventIntervalGen
from airlift.envs.events.malfunction_handler import MalfunctionHandler
from airlift.tests.util import generate_environment
from gym import logger
from airlift.solutions.baselines import RandomAgent
logger.set_level(logger.WARN)


def test_malfunction(render):
    seed = 546435
    np_random, seed = seeding.np_random(seed)
    malfunction_generator = EventIntervalGenerator(10, 11)  # rate, min, max
    malfunction_generator.seed(seed=1937)

    handler = MalfunctionHandler(malfunction_generator)
    env = generate_environment(num_of_airports=5, num_of_agents=1, malfunction_generator=malfunction_generator, poisson_lambda=.05)

    # Direct test to ParamMalfunctionGen
    malfunction = malfunction_generator.generate()
    assert malfunction.num_broken_steps != 0

    # Steps broken is always +1 of max. Should we change it to just be max?
    assert malfunction.num_broken_steps == 10 or malfunction.num_broken_steps == 11

    # Through Handler
    handler.step()
    assert handler.in_malfunction
    assert handler.num_malfunctions == 1

# This test isn't possible at the moment
def test_no_malfunction(render):
    seed = 546435
    np_random, seed = seeding.np_random(seed)
    malfunction_generator = NoEventIntervalGen()
    malfunction_generator.seed(seed=1937)
    handler = MalfunctionHandler(malfunction_generator)
    env = generate_environment(num_of_airports=3, num_of_agents=1, malfunction_generator=malfunction_generator, poisson_lambda=0)

    # Direct test to ParamMalfunctionGen
    malfunction = malfunction_generator.generate()
    assert malfunction.num_broken_steps == 0

    # Through Handler
    handler.step()
    assert not handler.in_malfunction
    # Fails at the moment, increments at any point a malfunction is attempted to be generated.
    assert handler.num_malfunctions == 0


def test_route_malfunction(render):
    malfunction_generator = EventIntervalGenerator(99, 100) # Always malfunction
    env = generate_environment(num_of_airports=5, num_of_agents=1, malfunction_generator=malfunction_generator, poisson_lambda=100, cargo_generator=StaticCargoGenerator(1))
    obs = env.reset(seed=54554455555)
    solution = RandomAgent()
    solution.reset(env.observe(), seed=34)
    actions = solution.policies(env.observe(), env.dones)
    obs, rewards, dones, _ = env.step(actions)
    observe_agent = env.observe('a_0')
    avail_routes = observe_agent['available_routes']
    # Available routes should be empty
    assert not len(avail_routes)

    # Check specific edges
    edge_one = (1, 2)
    edge_two = (2, 1)
    route = observe_agent['globalstate']['route_map'][observe_agent['plane_type']]
    edge_one_data = route.get_edge_data(*edge_one)
    edge_two_data = route.get_edge_data(*edge_two)
    assert not edge_one_data['route_available']
    assert not edge_two_data['route_available'] # this is broken?

    assert edge_one_data['mal'] == 100

    # Take a step, check decrement of malfunction timer
    actions = solution.policies(env.observe(), env.dones)
    obs, rewards, dones, _ = env.step(actions)
    observe_agent = env.observe('a_0')
    route = observe_agent['globalstate']['route_map'][observe_agent['plane_type']]
    edge_one_data = route.get_edge_data(*edge_one)
    edge_two_data = route.get_edge_data(*edge_two)
    assert edge_one_data['mal'] == 99
    assert edge_two_data['mal'] == 99

    # Check other steps
    for i in range(99):
        actions = solution.policies(env.observe(), env.dones)
        obs, rewards, dones, _ = env.step(actions)
        route = observe_agent['globalstate']['route_map'][observe_agent['plane_type']]
        edge_one_data = route.get_edge_data(*edge_one)
        edge_two_data = route.get_edge_data(*edge_two)
        assert edge_one_data['mal'] == 99 - (i+1)
        assert edge_two_data['mal'] == 99 - (i+1)

        # Are we done malfunctioning? Test to see if route is available
        if i == 99:
            assert edge_one_data['route_available']
            assert edge_two_data['route_available']
