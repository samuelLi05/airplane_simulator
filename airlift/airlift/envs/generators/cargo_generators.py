from typing import List, Collection, NamedTuple, Tuple
import networkx as nx
from gym.utils import seeding
from ordered_set import OrderedSet
from airlift.envs.airport import Airport, AirportID
from airlift.envs.cargo import Cargo, CargoID
from airlift.envs.events.event_generator import EventGenerator
from airlift.envs.route_map import RouteMap
from gym import logger
from airlift.utils.seeds import generate_seed


class CargoGenerator:
    """
    Handles the generation of Cargo Tasks
    """

    def __init__(self, num_initial_tasks, max_cargo_per_episode, max_stagger_steps=0, max_weight=1):
        self._np_random = None
        self.routemap = None
        self.num_initial_tasks = num_initial_tasks
        self.max_cargo_per_episode = max_cargo_per_episode
        self.current_cargo_count = 0
        self.max_weight = max_weight
        self.max_stagger_steps = max_stagger_steps

    def reset(self, routemap: RouteMap):
        self.routemap = routemap
        self.current_cargo_count = 0

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate_initial_orders(self) -> set:
        raise NotImplementedError

    def generate_dynamic_orders(self, elapsed_steps, max_cycles) -> List[Cargo]:
        return []

    def will_generate_more_cargo(self):
        return self.current_cargo_count < self.max_cargo_per_episode

    def generate_cargo_weight(self):
        """
        Generates a random weight for a cargo.
        """
        if self.max_weight > 1:
            return self._np_random.randint(1, self.max_weight)
        else:
            return 1


class StaticCargoGenerator(CargoGenerator):
    """
    Handles the generation of static cargo tasks. These are initialized upon the creation of the environment.
    """

    def __init__(self, num_of_tasks=1, soft_deadline_multiplier=50, hard_deadline_multiplier=100, max_stagger_steps=100, max_weight=1):
        super().__init__(num_of_tasks, num_of_tasks, max_weight=max_weight)

        self.max_stagger_steps = max_stagger_steps
        self._processing_time = None
        self._graph = None
        self.soft_deadline_multiplier = soft_deadline_multiplier
        self.hard_deadline_multiplier = hard_deadline_multiplier
        self.avg_hops = None
        self.avg_flighttime = None

    def reset(self, routemap: RouteMap):
        super().reset(routemap)
        self._processing_time = routemap.airports[0].processing_time  # Assume all processing times are the same

        self.avg_hops = nx.average_shortest_path_length(routemap.multigraph, weight=None) - 1
        self.avg_flighttime = nx.average_shortest_path_length(routemap.multigraph, weight="time")

    def generate_initial_orders(self) -> list:
        """
        Generates static cargo orders upon creation of the environment.

        :return: `order_list` - Returns a set that contains all the (static) generated cargo orders

        """

        self.current_cargo_count = self.num_initial_tasks
        cargo_list = []
        for i in range(self.num_initial_tasks):
            if self.max_stagger_steps != 0:
                stagger_duration = self._np_random.randint(0, self.max_stagger_steps)
            else:
                stagger_duration = self.max_stagger_steps
            cargo_list.append(self.generate_cargo_order(i, self.routemap.drop_off_airports, self.routemap.pick_up_airports, time_available=stagger_duration))
        return cargo_list

    def generate_dynamic_orders(self, elapsed_steps, max_cycles) -> List[Cargo]:
        return []

    def generate_cargo_order(self, cargo_id, drop_off_airports: OrderedSet[Airport],
                             pickup_airports: OrderedSet[Airport], time_available=0,
                             soft_deadline_multiplier=None, hard_deadline_multiplier=None) -> Cargo:
        """
        Generates cargo orders based on several parameters. Takes into account if we are running a scenario with a
        concentrated drop off or pick up location as well as non-concentrated locations that utilize the entire map.
        The dynamic cargo generator accesses this function without going through the above generate_order function.

        :parameter cargo_id: Incremental count of what cargo to create
        :parameter drop_off_airports: List that contains all the airports that are drop off locations'
        :parameter pick_up_airports: List that contains all the airports that are pick up locations
        :return: `cargo_task` : A fully generated cargo task

        """

        destination_airport = self._np_random.choice(drop_off_airports)
        source_airport = self._np_random.choice(pickup_airports - {destination_airport})
        soft_deadline, hard_deadline = self.create_schedule(source_airport, destination_airport,
                                                            soft_deadline_multiplier, hard_deadline_multiplier)

        hard_deadline = hard_deadline + time_available
        soft_deadline = soft_deadline + time_available

        assert soft_deadline > 0
        assert hard_deadline > 0
        cargo_task = Cargo(cargo_id,
                           source_airport,
                           destination_airport,
                           self.generate_cargo_weight(),
                           soft_deadline,
                           hard_deadline,
                           earliest_pickup_time=time_available)
        source_airport.add_cargo(cargo_task)
        return cargo_task

    def create_schedule(self, source, destination, soft_deadline_multiplier=None, hard_deadline_multiplier=None) -> \
    Tuple[int, int]:
        """
        Creates a schedule that cargo must be delivered by for it to be considered late (soft deadline) or completely
        missed (hard deadline)

        :parameter source: Source Airport
        :parameter destination: Destination Airport
        :return: `(soft_deadline, hard_deadline)` : Returns a tuple[int, int] containing soft/hard deadlines for cargo.

        """

        if soft_deadline_multiplier is None:
            soft_deadline_multiplier = self.soft_deadline_multiplier
        if hard_deadline_multiplier is None:
            hard_deadline_multiplier = self.hard_deadline_multiplier

        # Note for hops - we subtract indices to ignore the starting airport

        shortest_path_hops = nx.shortest_path_length(self.routemap.multigraph, source.id, destination.id,
                                                     weight=None) - 1
        shortest_path_flighttime = nx.shortest_path_length(self.routemap.multigraph, source.id, destination.id,
                                                           weight="time")

        # shortest_path_hops = nx.shortest_path_length(self._graph, source.id, destination.id, weight=None) - 1
        # shortest_path_flighttime = nx.shortest_path_length(self._graph, source.id, destination.id, weight="time")

        # Assume the return trip is about the same as the delivery trip (based on structure of dropoff/pickup zones)
        # We add one to processing time to account for the time step required to take off
        unit_deadline = 2 * ((self._processing_time + 1) * shortest_path_hops + shortest_path_flighttime)

        soft_deadline = round(soft_deadline_multiplier * unit_deadline)
        hard_deadline = round(hard_deadline_multiplier * unit_deadline)

        return soft_deadline, hard_deadline


class DynamicCargoGenerator(StaticCargoGenerator):
    """
    Handles the dynamic generation of cargo using the Poisson process as utilized by Flatlands for train malfunctions
    """

    def __init__(self, cargo_creation_rate: float, max_cargo_to_create: int, num_initial_tasks: int,
                 soft_deadline_multiplier=1,
                 hard_deadline_multiplier=1.5,
                 dynamic_soft_deadline_multiplier=None,
                 dynamic_hard_deadline_multiplier=None,
                 max_stagger_steps=100,
                 max_weight=1):
        super().__init__(num_of_tasks=num_initial_tasks,
                         soft_deadline_multiplier=soft_deadline_multiplier,
                         hard_deadline_multiplier=hard_deadline_multiplier,
                         max_stagger_steps=max_stagger_steps,
                         max_weight=max_weight)

        if dynamic_soft_deadline_multiplier is None:
            dynamic_soft_deadline_multiplier = soft_deadline_multiplier
        if dynamic_hard_deadline_multiplier is None:
            dynamic_hard_deadline_multiplier = hard_deadline_multiplier
        self.dynamic_soft_deadline_multiplier = dynamic_soft_deadline_multiplier
        self.dynamic_hard_deadline_multiplier = dynamic_hard_deadline_multiplier

        self.cargo_creation_rate = cargo_creation_rate
        self.max_cargo_per_episode = max_cargo_to_create + num_initial_tasks
        self.eventgen = EventGenerator(self.cargo_creation_rate)

    def generate_dynamic_orders(self, elapsed_steps, max_cycles) -> List[Cargo]:
        """
        The EventGenerator controls whether a new dynamic cargo is generated or not using poisson distribution. Makes sure that
        the upperbound limit of max cargo generated per episode is also adhered to.

        :return: `cargo_task` : a List that contains a single Cargo or an empty list if no cargo was generated

        """

        # If we are up to 90% finished with max_cycles, stop generating more dynamic cargo.
        # Cargo may be generated that is impossible to complete.
        threshold_reached = elapsed_steps > (max_cycles * .9)

        if self.eventgen.generate() and \
                self.current_cargo_count < self.max_cargo_per_episode and not threshold_reached:
            cargo_task = self.generate_cargo_order(self.current_cargo_count,
                                                   self.routemap.drop_off_airports,
                                                   self.routemap.pick_up_airports,
                                                   soft_deadline_multiplier=self.dynamic_soft_deadline_multiplier,
                                                   hard_deadline_multiplier=self.dynamic_hard_deadline_multiplier,
                                                   time_available=elapsed_steps)
            self.current_cargo_count += 1
            logger.info("Cargo Generated " + "Current " + str(self.current_cargo_count) + " Max: " + str(
                self.max_cargo_per_episode))

            return [cargo_task]

        return []

    def reset(self, routemap):
        super().reset(routemap)

    def seed(self, seed=None):
        super().seed(seed)
        self.eventgen.seed(seed=generate_seed(self._np_random))


class EarliestPickupCargoGenerator(StaticCargoGenerator):

    def __init__(self, num_of_tasks=1, soft_deadline_multiplier=1, hard_deadline_multiplier=1.5, time_step_interval=1,
                 min_range=0, max_range=15000):
        super().__init__(num_of_tasks=num_of_tasks,
                         soft_deadline_multiplier=soft_deadline_multiplier,
                         hard_deadline_multiplier=hard_deadline_multiplier)

        self.time_table = []
        self._processing_time = None
        self._graph = None
        self.soft_deadline_multiplier = soft_deadline_multiplier
        self.hard_deadline_multiplier = hard_deadline_multiplier
        self.avg_hops = None
        self.avg_flighttime = None

        # Create a list of intervals
        self.time_step_interval = time_step_interval
        self.max_range = max_range
        self.min_range = min_range

    def create_discrete_timetable(self, start, end, time_step_interval):
        current = start
        while current < end:
            yield current
            current += self._np_random.choice(time_step_interval)

    def populate_time_table(self):
        self.time_table = [num for num in
                           self.create_discrete_timetable(self.min_range, self.max_range,
                                                          self.time_step_interval)]
        self.time_table.reverse()

    def reset(self, routemap):
        super().reset(routemap)

    def seed(self, seed=None):
        super().seed(seed)

    def generate_cargo_order(self, cargo_id, drop_off_airports: OrderedSet[Airport],
                             pickup_airports: OrderedSet[Airport], time_available=0) -> Cargo:

        # Populate time table for cargo availability
        if not self.time_table:
            self.populate_time_table()

        time_available = self.time_table.pop()
        destination_airport = self._np_random.choice(drop_off_airports)
        source_airport = self._np_random.choice(pickup_airports - {destination_airport})
        soft_deadline, hard_deadline = self.create_schedule(source_airport, destination_airport)
        cargo_task = Cargo(cargo_id,
                           source_airport,
                           destination_airport,
                           self.generate_cargo_weight(),
                           # Add time_available to the deadlines
                           soft_deadline + time_available,
                           hard_deadline + time_available,
                           earliest_pickup_time=time_available)
        source_airport.add_cargo(cargo_task)
        return cargo_task


class CargoInfo(NamedTuple):
    """Primarily used in testing purposes for the hard coded cargo generator. Properties resemble that of the Cargo class"""
    id: CargoID
    source_airport_id: AirportID
    end_airport_id: AirportID
    weight: float = 1
    soft_deadline: int = 2 ** 32
    hard_deadline: int = 2 ** 32
    time_available: int = 0


class HardcodedCargoGenerator(CargoGenerator):
    """
    A basic cargo generator that allows for the initialization of cargo using pre-determined values instead of the
    static or dynamic generators. Generally used for testing purposes.
    """

    def __init__(self, initialcargoinfo=None):
        super().__init__(len(initialcargoinfo), len(initialcargoinfo))
        self._initialcargoinfo: Collection[CargoInfo] = initialcargoinfo
        self.soft_deadline_multiplier = None
        self.hard_deadline_multiplier = None

    def generate_initial_orders(self) -> set:
        """
        Generates the initial hard-coded cargo values.

        :return: `initialcargo` : A set that contains the cargo.

        """
        initialcargo = [Cargo(c.id,
                              self.routemap.airports_by_id[c.source_airport_id],
                              self.routemap.airports_by_id[c.end_airport_id],
                              c.weight,
                              c.soft_deadline,
                              c.hard_deadline,
                              c.time_available)
                        for c in self._initialcargoinfo]

        for cargo in initialcargo:
            cargo.source_airport.add_cargo(cargo)

        self.current_cargo_count = len(initialcargo)
        return set(initialcargo)


class NoCargoGenerator(CargoGenerator):
    """
    Initialize an environment with no cargo. Generally used for testing purposes.
    """

    def __init__(self):
        super().__init__(0, 0)
        self.soft_deadline_multiplier = None
        self.hard_deadline_multiplier = None

    def generate_initial_orders(self) -> set:
        return set()
