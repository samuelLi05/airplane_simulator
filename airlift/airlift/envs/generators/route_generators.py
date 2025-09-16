import itertools
import warnings
from collections import Counter
from math import ceil
from typing import NamedTuple, Collection

import gym
import networkx as nx
from gym.utils import seeding
from ordered_set import OrderedSet

from airlift.envs.plane_types import PlaneTypeID
from airlift.envs.airport import AirportID
from airlift.envs.events.event_interval_generator import NoEventIntervalGen, EventIntervalGenerator
from airlift.envs.route_map import RouteMap
from airlift.utils.seeds import generate_seed


class RouteGenerator:
    def __init__(self,
                 malfunction_generator,
                 route_ratio=None,
                 poisson_lambda=0.0):  # Ratio of edges to nodes (roughly sets degree of the nodes)
        self.malfunction_generator = malfunction_generator
        self.route_ratio: int = route_ratio
        self.poisson_lambda = poisson_lambda

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.malfunction_generator.seed(seed=generate_seed(self._np_random))

    def generate(self, routemap: RouteMap):
        raise NotImplementedError

    def _add_routes_at_random(self, routemap, routetuples, bidirectional=True):
        if self.route_ratio is None:
            chosen_routetuples = routetuples
        else:
            num_airports = len(routemap.airports)
            num_routes = round(self.route_ratio * num_airports)
            # chosen_routetuples = self._np_random.choice(routetuples, min(num_routes, len(routetuples)))

            # Add an edge for one airport at a time.
            # Choose airports uniformly at random to try and balance the degrees of the nodes (to prevent giving preference to nodes with more edges)
            # First, make a dictionary to track the edges of each node
            routes_by_airport = {a: [] for a in routemap.airports}
            for r in routetuples:
                routes_by_airport[r[1]].append(r)
                routes_by_airport[r[2]].append(r)

            # Choose airports at random
            chosen_routetuples = []
            airport_choices = self._np_random.choice(routemap.airports, min(num_routes, len(routetuples)))
            for a1 in airport_choices:
                # If there isn't an edge for this node, make a new choice until we find one
                # Limit # of retries, just to make sure we don't hang
                trycount = 0
                while not routes_by_airport[a1] and trycount < 1000:
                    a1 = self._np_random.choice(routemap.airports)
                    trycount += 1

                routeidx = self._np_random.choice(
                    len(routes_by_airport[a1]))  # Sampling directly from the tuple list has issues...
                route = routes_by_airport[a1][routeidx]
                routes_by_airport[route[1]].remove(route)
                routes_by_airport[route[2]].remove(route)  # Remove the route from the airport at the other end
                chosen_routetuples.append(route)

        for routetuple in chosen_routetuples:
            routemap.add_route(plane=routetuple[0],
                               start=routetuple[1],
                               end=routetuple[2],
                               time=routetuple[3],
                               cost=routetuple[4],
                               malfunction_generator=routetuple[5],
                               bidirectional=bidirectional)

    def _connect_components(self, routemap, malfunction_generator):
        # From https://stackoverflow.com/questions/70631490/how-can-i-make-np-argmin-code-without-numpy
        def argmin(a):
            return min(range(len(a)), key=lambda x: a[x])

        while True:
            # Get components. Quit when there is just one remaining.
            components = list(nx.strongly_connected_components(routemap.multigraph))
            if len(components) > 1:
                gym.logger.warn("Route map contains multiple connected components - performing join...")
            else:
                break

            # nodes1 = nodes in largest component
            # nodes2 = nodes in remaining components
            components = sorted(components, key=len)
            airports1 = routemap.airports_by_ids(components[-1])
            airports2 = routemap.airports_by_ids(itertools.chain(*components[:-1]))

            # Find distances between airports from each set and pick the airports that are closest together
            airport_pairs = list(itertools.product(airports1, airports2))
            distances = [routemap.map.distance(a1.position, a2.position) for a1, a2 in airport_pairs]
            a1, a2 = airport_pairs[argmin(distances)]

            # Pick the plane type based on the existing routes connected to these airports.
            # We choose the plane type with the most existing routes.
            edges = list(routemap.multigraph.edges(a1.id, keys=True)) + list(
                routemap.multigraph.edges(a2.id, keys=True))
            counts = Counter([e[2] for e in edges])
            if counts:
                plane_type = counts.most_common(1)[0][0]
            else:
                plane_type = self._np_random.choice(routemap.plane_types)

            # Add the route
            distance = routemap.map.distance(a1.position, a2.position)
            routemap.add_route(plane=plane_type,
                               start=a1,
                               end=a2,
                               time=ceil(distance / plane_type.speed),
                               cost=distance,
                               malfunction_generator=malfunction_generator,
                               bidirectional=True)


class RouteByDistanceGenerator(RouteGenerator):
    def __init__(self,
                 malfunction_generator=NoEventIntervalGen(),
                 route_ratio=None, poisson_lambda=0.0,
                 ):
        super().__init__(malfunction_generator, route_ratio, poisson_lambda=poisson_lambda)

    def generate(self, routemap: RouteMap):
        """
        Creates edges based on an airplane models maximum range.

        :Parameters:
        ----------
        `routemap` : a RouteMap that contains all the DiGraphs in a dictionary.

        """
        routetuples = []
        for plane_type in routemap.plane_types:
            for a1, a2 in itertools.combinations(routemap.airports, 2):
                distance = routemap.map.distance(a1.position, a2.position)
                if 0 < distance <= plane_type.max_range:
                    routetuples.append(
                        (plane_type, a1, a2, ceil(distance / plane_type.speed), distance, self.malfunction_generator))

        self._add_routes_at_random(routemap, routetuples, bidirectional=True)
        self._connect_components(routemap, self.malfunction_generator)
        routemap.set_poisson_params(self.poisson_lambda)
        assert nx.is_strongly_connected(routemap.multigraph), "Route map is not strongly connected"


# Plane id 0 is the large long-haul plane
# Plane id 1 is the small short-haul plane
class LimitedDropoffEntryRouteGenerator(RouteGenerator):
    def __init__(self,
                 malfunction_generator=NoEventIntervalGen(),
                 route_ratio=None,
                 drop_off_fraction_reachable=0,
                 pick_up_fraction_reachable=0,
                 poisson_lambda=0.0):

        super().__init__(malfunction_generator, route_ratio)
        self.drop_off_fraction_reachable: float = drop_off_fraction_reachable
        self.pick_up_fraction_reachable: float = pick_up_fraction_reachable
        self.poisson_lambda = poisson_lambda

    def generate(self, routemap: RouteMap):
        # Assume there are two plane types with id's 0 and 1
        assert {p.id for p in routemap.plane_types} == {0, 1}

        drop_off_reachable = {a for a in routemap.drop_off_airports if
                              self._np_random.binomial(1, self.drop_off_fraction_reachable)}
        pick_up_reachable = {a for a in routemap.pick_up_airports if
                             self._np_random.binomial(1, self.pick_up_fraction_reachable)}

        routetuples = []
        for a1, a2 in itertools.combinations(routemap.airports, 2):
            for plane_type in routemap.plane_types:
                valid_route = True
                if plane_type.id == 0:
                    if (a1.in_drop_off_area and a1 not in drop_off_reachable) or \
                            (a2.in_drop_off_area and a2 not in drop_off_reachable) or \
                            (a1.in_pick_up_area and a1 not in pick_up_reachable) or \
                            (a2.in_pick_up_area and a2 not in pick_up_reachable):
                        valid_route = False
                elif plane_type.id == 1:
                    if not a1.in_drop_off_area and \
                            not a2.in_drop_off_area and \
                            not a1.in_pick_up_area and \
                            not a2.in_pick_up_area:
                        valid_route = False
                else:
                    assert False, "Invalid plane type"

                if valid_route:
                    distance = routemap.map.distance(a1.position, a2.position)
                    if 0 < distance <= plane_type.max_range:
                        routetuples.append((plane_type, a1, a2, ceil(distance / plane_type.speed), distance,
                                            self.malfunction_generator))

        self._add_routes_at_random(routemap, routetuples, bidirectional=True)
        self._connect_components(routemap, self.malfunction_generator)
        routemap.set_poisson_params(self.poisson_lambda)
        assert nx.is_strongly_connected(routemap.multigraph), "Route map is not strongly connected"



class RouteInfo(NamedTuple):
    start: AirportID
    end: AirportID
    plane_id: PlaneTypeID = 0
    cost: int = 1
    time: int = 1
    malfunction_generator: EventIntervalGenerator = NoEventIntervalGen()


class HardcodedRouteGenerator(RouteGenerator):
    """
    Can be utilized to hard code a route between nodes. This class is primarily used for testing purposes.
    """
    def __init__(self, routeinfo):
        super().__init__(NoEventIntervalGen())
        self.routeinfo: Collection[RouteInfo] = routeinfo

    def generate(self, routemap: RouteMap):
        """Adds edges to nodes based upon the information passed to it using the initializer"""
        for r in self.routeinfo:
            routemap.add_route(routemap.plane_types_by_id[r.plane_id],
                               routemap.airports_by_id[r.start],
                               routemap.airports_by_id[r.end],
                               time=r.time,
                               cost=r.cost,
                               malfunction_generator=r.malfunction_generator,
                               bidirectional=True)
        routemap.set_poisson_params(self.poisson_lambda)

        if not nx.is_strongly_connected(routemap.multigraph):
            warnings.warn("Route map is not strongly connected")