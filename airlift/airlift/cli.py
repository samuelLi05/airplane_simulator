import time
from gym import logger

from airlift.envs.plane_types import PlaneType
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator, LimitedDropoffEntryRouteGenerator
from airlift.envs.generators.map_generators import PerlinMapGenerator
from airlift.envs.generators.world_generators import AirliftWorldGenerator
import click

from airlift.envs.renderer import FlatRenderer
from airlift.solutions.baselines import ShortestPath, RandomAgent

logger.set_level(logger.INFO)


def create_env(showroutes=False):
    multiple_plane_types = [PlaneType(id=0, max_range=1, speed=0.05, max_weight=20),
                            PlaneType(id=1, max_range=1, speed=0.05, max_weight=10)]
    return AirliftEnv(
        AirliftWorldGenerator(
            plane_types=multiple_plane_types,
            airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
                                                     max_airports=30,
                                                     num_drop_off_airports=6,
                                                     num_pick_up_airports=6,
                                                     processing_time=10,
                                                     working_capacity=2,
                                                     airports_per_unit_area=2),
            route_generator=LimitedDropoffEntryRouteGenerator(
                malfunction_generator=EventIntervalGenerator(
                    min_duration=10,
                    max_duration=20),
                route_ratio=2,  # math.floor(math.log(num_airports)/2),
                drop_off_fraction_reachable=0,
                pick_up_fraction_reachable=0,
                poisson_lambda=0.2),
            cargo_generator=DynamicCargoGenerator(cargo_creation_rate=1 / 100,
                                                  soft_deadline_multiplier=40,
                                                  hard_deadline_multiplier=120,
                                                  num_initial_tasks=40,
                                                  max_cargo_to_create=10),
            airplane_generator=AirplaneGenerator(10), max_cycles=5000
        ),
        renderer=FlatRenderer(show_routes=showroutes)
    )


@click.command()
@click.option('--frame_pause_time', default=0.01, help='Number of seconds to pause between frames')
@click.option('--showroutes', is_flag=True, show_default=True, default=True,
              help="Show routes on the environment rendering")
def demo(frame_pause_time, showroutes):
    solution = ShortestPath()
    env = create_env(showroutes)
    _done = False
    obs = env.reset(seed=325)  # 365  372
    solution.reset(obs, seed=4594454)
    while not _done:
        # Compute Action
        actions = solution.policies(env.observe(), env.dones)
        obs, rewards, dones, _ = env.step(actions)  # If there is no observation, just return 0
        _done = all(dones.values())
        env.render()
        if frame_pause_time > 0:
            time.sleep(frame_pause_time)


if __name__ == "__main__":
    demo()
