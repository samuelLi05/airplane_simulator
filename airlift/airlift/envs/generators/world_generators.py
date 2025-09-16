from typing import Tuple, List, Set, Optional
from gym.utils import seeding

from airlift.envs.agents import EnvAgent
from airlift.envs.airport import Airport
from airlift.envs.cargo import Cargo
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.routemap_updater import NoRouteMapUpdater
from airlift.envs.route_map import RouteMap
from airlift.utils.seeds import generate_seed
from airlift.envs.plane_types import PlaneType


class WorldGenerator:
    """
    The base class for generating the airlift world.
    """

    def __init__(self, max_cycles=2 ** 32):
        self.max_cycles = max_cycles
        self._np_random = None

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate(self) -> Tuple[RouteMap, List[EnvAgent], Set[Cargo]]:
        raise NotImplementedError


class AirliftWorldGenerator(WorldGenerator):
    """
    Utilizes all the generators (Airport, Route, Cargo, Airplane etc...) to create the environment.
    """
    def __init__(self,
                 plane_types=[PlaneType(id=0, model='A0', max_range=2.0, speed=0.1, max_weight=5)],
                 airport_generator=RandomAirportGenerator(3),
                 route_generator=RouteByDistanceGenerator(),
                 cargo_generator=StaticCargoGenerator(1),
                 airplane_generator=AirplaneGenerator(1),
                 routemap_updater=NoRouteMapUpdater(),
                 static_airports=False,
                 max_cycles=10 ** 4):
        super().__init__(max_cycles=max_cycles)

        self.plane_types = plane_types
        self.airport_generator = airport_generator
        self.route_generator = route_generator
        self.cargo_generator = cargo_generator
        self.airplane_generator = airplane_generator
        self.static_airports = static_airports

        if not isinstance(routemap_updater, list):
            routemap_updater = [routemap_updater]
        self.routemap_updater = routemap_updater

        self._np_random = None

        self._airports: Optional[List[Airport]] = None
        self._drop_off_area = None
        self._pick_up_area = None
        self._map = None


    def seed(self, seed=None):
        """
        Upon the initialization of the environment all the generators will be seeded.
        """
        super().seed(seed)
        self.airport_generator.seed(generate_seed(self._np_random))
        self.route_generator.seed(generate_seed(self._np_random))
        self.cargo_generator.seed(generate_seed(self._np_random))
        self.airplane_generator.seed(generate_seed(self._np_random))
        for u in self.routemap_updater:
            u.seed(generate_seed(self._np_random))
        self._map = None
        self.airports = None

    def generate(self) -> Tuple[RouteMap, List[EnvAgent], Set[Cargo]]:
        """
        Generates the initial cargo orders, the routes for each plane type, environment agents (airplanes). The world map is also generated within
        the airport_generator.generate() function. One of the initialization parameters for the airport generator is to decide
        which world map generator to use with the keyword mapgen="".

        :return: `(routemap, airplanes, cargo)` : Returns a Tuple that contains the Routemap, the list of agents and the set of cargo generated.

        """
        # Re-generate map as necessary
        if not self.static_airports or self.airports is None:
            self._airports, self._map, self._drop_off_area, self._pick_up_area = self.airport_generator.generate()

        route_map = RouteMap(self._map, self._airports, self.plane_types, self._drop_off_area, self._pick_up_area)

        self.route_generator.generate(route_map)

        self.cargo_generator.reset(routemap=route_map)
        cargo = self.cargo_generator.generate_initial_orders()

        airplanes = self.airplane_generator.generate(route_map)

        for u in self.routemap_updater:
            u.update(route_map)

        # Seed the RouteMap and the poisson distribution
        route_map.seed(generate_seed(self._np_random))
        route_map.poisson_dist.seed(generate_seed(self._np_random))

        return route_map, \
               airplanes, \
               cargo

    # Various properties which do not change (these can be accessed before seeding/generation)
    @property
    def num_agents(self):
        return self.airplane_generator.num_agents

    @property
    def max_airports(self):
        return self.airport_generator.max_airports

    @property
    def max_cargo_per_episode(self):
        return self.cargo_generator.max_cargo_per_episode

    @property
    def soft_deadline_multiplier(self):
        return self.cargo_generator.soft_deadline_multiplier

    @property
    def hard_deadline_multiplier(self):
        return self.cargo_generator.hard_deadline_multiplier

    @property
    def route_malfunction_rate(self):
        return self.route_generator.malfunction_generator.malfunction_rate

    @property
    def route_malfunction_max_duration(self):
        return self.route_generator.malfunction_generator.max_duration

    @property
    def route_malfunction_min_duration(self):
        return self.route_generator.malfunction_generator.min_duration