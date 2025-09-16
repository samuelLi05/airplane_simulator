import random
from collections import Counter
from functools import reduce
from math import ceil

import networkx as nx
import numpy as np
from gym.utils import seeding

from airlift.utils.seeds import generate_seed


class RouteMapUpdater:
    def __init__(self):
        self._np_random = None

    def update(self, routemap):
        raise NotImplementedError

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)


def get_all_shortest_paths(routemap):
    # Get the airports that are pick up and drop zones
    drop_off_airports = [airport for airport in routemap.airports if airport.in_drop_off_area]
    pick_up_airports = [airport for airport in routemap.airports if airport.in_pick_up_area]
    result = list(zip(drop_off_airports, pick_up_airports))

    # Select a value we will count up to or more
    num_pick_up_and_drop_off = len(drop_off_airports) + len(pick_up_airports)
    count_value = ceil(num_pick_up_and_drop_off * .4)

    # Get the shortest path
    all_shortest_paths = [(nx.shortest_path(routemap.multigraph, source=airport[0].id, target=airport[1].id)) for
                          airport in result]
    return all_shortest_paths, count_value

class VaryCapacityRouteMapUpdater(RouteMapUpdater):
    def __init__(self):
        super().__init__()

    def update(self, routemap):
        all_shortest_paths, count_value = get_all_shortest_paths(routemap)


        # flatten the list to find the counts
        flattened_list = [item for sublist in all_shortest_paths for item in sublist]
        counter = Counter(flattened_list)
        most_common_values = [(value, count) for value, count in counter.items() if count >= count_value]
        airports = routemap.airports_by_ids([value[0] for value in most_common_values])

        # Update the capacity to 1 in airports that repeat >= count_value
        for airport in airports:
            routemap.update_airport_capacity(airport.id, 1)

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)


class RandomAirportWorkingCapacities(RouteMapUpdater):
    def __init__(self, capacities, probs):
        super().__init__()
        self.working_capacities = capacities
        self.probs = probs

    def update(self, routemap):
        for airport in routemap.airports:
            routemap.update_airport_capacity(airport.id,  self._np_random.choice(self.working_capacities, size=1, p=self.probs)[0])

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)


class SprinkleMalfunction_RouteMapUpdater(RouteMapUpdater):
    def __init__(self, malfunction_generators, probs=None):
        super().__init__()
        self.malfunction_generators = malfunction_generators
        if probs is None:
            n = len(malfunction_generators)
            probs = [1 / n] * n  # Uniform probability distribution over all malfunctions
        self.probs = probs

    def seed(self, seed=None):
        super().seed(seed)
        for m in self.malfunction_generators:
            m.seed(generate_seed(
                self._np_random))  # Note we may re-seed a malfunction generator already seeded by route generator. This is OK.

    def update(self, routemap):
        G = routemap.multigraph
        for u, v, planetype, d in G.edges(keys=True, data=True):
            if u < v:
                m = self._np_random.choice(self.malfunction_generators, p=self.probs)
                d["mal"].malfunction_generator = m
                G.edges[v, u, planetype]["mal"].malfunction_generator = m


class MalfunctionAlongshortestPath_RouteMapUpdater(RouteMapUpdater):
    def __init__(self, malfunction_generator_hard, probs=None):
        super().__init__()
        self.malfunction_generator_hard = malfunction_generator_hard

    def seed(self, seed=None):
        super().seed(seed)
        self.malfunction_generator_hard.seed(generate_seed(self._np_random))

    def update(self, routemap):
        G = routemap.multigraph
        m = self.malfunction_generator_hard
        all_shortest_paths = get_all_shortest_paths(routemap)
        for path in all_shortest_paths:
            for u, v in zip(path, path[1:]):
                for planetype, data in G.get_edge_data(u, v).items():
                    data["mal"].malfunction_generator = m
                    G.edges[v, u, planetype]["mal"].malfunction_generator = m


class NoRouteMapUpdater(RouteMapUpdater):
    def __init__(self):
        super().__init__()

    def update(self, routemap):
        pass
