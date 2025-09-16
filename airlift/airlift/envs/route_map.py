import random
from typing import List, Collection, Set, Dict, Any
import networkx as nx
import numpy as np
from gym.utils import seeding

from airlift.envs.airport import Airport, AirportID
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.events.poisson_dist_gen import PoissonDistribution
from airlift.envs.world_map import Map, FlatArea
from airlift.envs.plane_types import PlaneType, PlaneTypeID
from airlift.envs.events.malfunction_handler import MalfunctionHandler
from ordered_set import OrderedSet

"""
The RouteMap class is used for the creation of a dictionary of DiGraphs (based on plane model) that assists
the environment in what routes an agent is able to take from one node to another.
"""


class RouteMap:
    """
    Defines the graph network between the Airports for each airplane model. Creates and controls all the edges
    as well as any malfunctions that may occur.
    """

    def __init__(self, map: Map, airports: List[Airport], plane_types: Collection[PlaneType] = None, drop_off_area=None,
                 pick_up_area=None):
        self.edges_in_malfunction = []
        self.poisson_dist = None
        self.event_generator = None
        self.previous_edges = None
        self.map = map
        self.graph: Dict[PlaneType, nx.DiGraph] = {}

        self.plane_types: List[PlaneType] = plane_types
        self.plane_types_by_id: Dict[PlaneTypeID, PlaneType] = {t.id: t for t in plane_types}

        self.airports: List[Airport] = airports
        self.airports_by_id: Dict[AirportID, Airport] = {a.id: a for a in airports}

        self.add_plane_models()
        self.add_vertices()
        self.max_routes_from_airport = len(airports)

        self.drop_off_area: FlatArea = drop_off_area
        self.pick_up_area: FlatArea = pick_up_area

        self.drop_off_airports: OrderedSet = OrderedSet(filter(lambda airport: airport.in_drop_off_area, airports))
        if not self.drop_off_airports:
            self.drop_off_airports = OrderedSet(airports)

        self.pick_up_airports: OrderedSet = OrderedSet(filter(lambda airport: airport.in_pick_up_area, airports))
        if not self.pick_up_airports:
            self.pick_up_airports = OrderedSet(airports)

        for plane in self.plane_types:
            self._route_malfunction = {(u, v,): d['mal']
                                       for u, v, d in self.graph[plane].edges(data=True)}

        self._multigraph: nx.MultiDiGraph = None
        self._malfunction_handlers = []
        self.all_edges = None

        self._np_random = None

    def set_poisson_params(self, lambda_value):
        self.poisson_dist = PoissonDistribution(lambda_value)

    def __repr__(self):
        return "Graph Data | Distance: " + " | Airplane Type: " + str(self.plane_types) + \
            " | Graph Information: " + str(self.graph) + " | Memory Address: " + hex(id(self))

    def airports_by_ids(self, ids: Collection[AirportID]) -> List[Airport]:
        return [self.airports_by_id[id] for id in ids]

    @staticmethod
    def build_multigraph(graph_dict) -> nx.MultiDiGraph():
        G = nx.MultiDiGraph()
        for k, digraph in graph_dict.items():  # Note that keys can be any data type
            for node, d in digraph.nodes(data=True):
                if node not in G.nodes:
                    G.add_node(node, **d)
            for u, v, d in digraph.edges(data=True):
                G.add_edge(u, v, k, **d)
        return G

    @property
    # Multigraph containing all planes
    def multigraph(self) -> nx.MultiDiGraph():
        if self._multigraph is None:
            self._multigraph = self.build_multigraph(self.graph)

        return self._multigraph

    def add_plane_models(self) -> None:
        """
        Creates a DiGraph for all airplane models and stores them in the self._graph dictionary
        """
        for plane in self.plane_types:
            self.graph[plane] = nx.DiGraph()

        self._multigraph = None

    def add_vertices(self) -> None:
        """
        Add nodes to a graph based on coordinate position
        """

        for plane in self.plane_types:
            for airport in self.airports:
                self.graph[plane].add_node(airport.id, pos=airport.position, airport=airport, working_capacity=airport.allowed_capacity)

        self._multigraph = None

    def add_route(self,
                  plane: PlaneType,
                  start: Airport,
                  end: Airport,
                  time: int,
                  cost: float,
                  malfunction_generator: EventIntervalGenerator,
                  bidirectional=True) -> None:
        """
        Adds an edge between two nodes. Populates the edge with airplane model, time, cost and adds a malfunction
        object to it.

        :parameter plane_model: Airplane Model
        :parameter airport1: From Airport
        :parameter airport2: To Airport
        :parameter time: Time to traverse the edge
        :parameter cost: Cost of traversing the edge
        :parameter malfunction_generator: Parameters for creating route malfunctions
        :parameter bidirectional: Boolean, True if route is bidirectional.

        """
        self.event_generator = malfunction_generator
        # Use the same handler for both routes so that both go offline at the same time
        handler = MalfunctionHandler(malfunction_generator)
        self._malfunction_handlers.append(handler)

        self.graph[plane].add_edge(start.id, end.id, cost=cost,
                                   plane_type=plane.id,
                                   time=time,
                                   mal=handler)
        if bidirectional:
            self.graph[plane].add_edge(end.id, start.id, cost=cost,
                                       plane_type=plane.id,
                                       time=time,
                                       mal=handler,
                                       expected_mal_steps=malfunction_generator.expected_mal_steps)

        self._multigraph = None

        assert (self.get_flight_cost(start, end, plane) > 0)
        assert (self.get_flight_time(start, end, plane) > 0)

    def update_airport_capacity(self, airport_id, allowed_capacity):
        self.airports_by_id[airport_id].allowed_capacity = allowed_capacity
        for plane in self.plane_types:
            self.graph[plane].nodes[airport_id]["working_capacity"] = allowed_capacity

    def draw_graph(self, plane_type_ids_to_draw=None, show_cap=False) -> None:
        import matplotlib.pyplot as plt
        EDGECOLORS = ['b', 'r', 'g']
        EDGESTYLES = ['-', ':', '--']

        if plane_type_ids_to_draw is None:
            plane_types_to_draw = self.plane_types
        else:
            if plane_type_ids_to_draw is not List:
                plane_type_ids_to_draw = [plane_type_ids_to_draw]
            plane_types_to_draw = [p for p in self.plane_types if p.id in plane_type_ids_to_draw]

        """
        Draws the graph using plyplot and displays it on the screen. Since our graphs origin is top left, we have to
        invert the y-axis for proper results.
        """
        for plane in plane_types_to_draw:
            pos = nx.get_node_attributes(self.graph[plane], 'pos')
            cap = nx.get_node_attributes(self.graph[plane], 'working_capacity')
            graph_edge_labels = dict([((u, v,), '{0} ({1})'.format(d['time'], d['cost']))
                                      for u, v, d in self.graph[plane].edges(data=True)])

            nx.draw_networkx_edge_labels(self.graph[plane], pos, edge_labels=graph_edge_labels,
                                         label_pos=0.5, font_size=7)

            if show_cap:
                nx.draw(self.graph[plane], pos, with_labels=True, edge_color=EDGECOLORS[plane.id],
                        style=EDGESTYLES[plane.id], labels={k: str(v) for k, v in cap.items()})
            else:
                nx.draw(self.graph[plane], pos, with_labels=True, edge_color=EDGECOLORS[plane.id],
                        style=EDGESTYLES[plane.id])
            # position = nx.get_edge_attributes(self._graph[plane.model], "time")

        # Change origin to top left
        plt.gca().invert_yaxis()
        plt.show()
        # plt.savefig("./airlift/png/graph.png")

    def get_metrics(self) -> None:
       cap = [list(nx.get_node_attributes(self.graph[plane], 'working_capacity').values()) for plane in self.plane_types]
       return min(min(cap)), max(max(cap))

    def step(self):
        """
        At each time step the routes are checked for malfunctions. If they are in a malfunction the route will become
        unavailable
        """
        # Generating this here or within the airliftenv probably doesn't make a difference
        if isinstance(self.event_generator, EventIntervalGenerator):
            num_events = self.poisson_dist.generate_events()
            selected_edges = self.select_random_edges_from_graph(num_events)

            # Keep a set of visited edges so that we don't step through bidirectional edges as well.
            # Both (1,2) and (2,1) have the same MalfunctionHandler object. Bidirectional edges are already taken care of in
            # the add_route function
            visited_edges = set()
            for edge in self._multigraph.edges(data=True):
                source, target, attributes = edge

                edge_tuple = tuple(sorted((source, target)))
                route = attributes['mal']

                # 1. Check if the edge is selected and hasn't been visited yet
                # 2. Or if it hasn't been selected but is in a malfunction and hasn't been visited.
                # This ensures we step through all edges that we need to place into a malfunction and also decrement
                # the malfunction down counter in those already offline without doing it twice.
                if (edge_tuple in selected_edges and edge_tuple not in visited_edges) or \
                        (route.in_malfunction and edge_tuple not in visited_edges):
                    route.step()

                    # Add the edge to visited
                    visited_edges.add(edge_tuple)

    def select_random_edges_from_graph(self, number_of_events):

        if self.all_edges is None:
            self.all_edges = list(self._multigraph.edges())

        if number_of_events > len(self.all_edges):
            number_of_events = len(self.all_edges)

        self._np_random.shuffle(self.all_edges)
        selected_edges = self.all_edges[:number_of_events]
        return selected_edges

    def get_flight_time(self, start: Airport, end: Airport, plane: PlaneType) -> int:
        """
        Gets the flight time associated with traversing from start to end Airport.

        :parameter start: Starting Airport
        :parameter end: Ending Airport
        :parameter plane: Airplane Model
        :return: `flight_time` : Flight Time associated with traversing from start to end node

        """
        return self.graph[plane].edges[start.id, end.id]["time"]

    def get_flight_cost(self, start: Airport, end: Airport, plane: PlaneType) -> float:
        """
        Gets the flight cost associated with traversing from start to end airport.

        :parameter start: Starting Airport
        :parameter end: Ending Airport
        :parameter plane: Airplane Model
        :return: `flight_distance` : Returns the distance (cost) associated with traversing from start to end node

        """
        return self.graph[plane].edges[start.id, end.id]["cost"]

    def reachable(self, source: Airport, destination: Airport, plane_type: PlaneType) -> bool:
        """
        Checks to see if an airport is reachable from the current airport.

        :parameter destination: Destination Airport
        :parameter source: Starting Airport
        :parameter plane_type: Airplane Model
        :return: `boolean` : Returns True if an airplane can travel directly from source to destination

        """

        return (source.id, destination.id) in self.graph[plane_type].edges

    def get_malfunction_time(self, source: AirportID, destination: AirportID, plane_type: PlaneType) -> int:
        """
        Returns the time remaining on a route being offline.

        :parameter source: Starting Airport
        :parameter destination: Destination Airport
        :parameter plane_type: Airplane Model
        :return: `int`: An integer that contains how long (in steps) an edge is malfunctioning for

        """
        return self.graph[plane_type].edges[(source, destination)]['mal'].malfunction_down_counter

    # Count for total malfunctions seems to be too high. Unsure what the cause is.
    def get_total_malfunctions(self) -> int:
        """
        Returns total malfunctions in episode.

        :return: `int` : An integer containing how many malfunctions have occurred

        """
        total_malfunctions_in_episode = 0
        for plane_type in self.plane_types:
            route_malfunction = dict([((u, v,), d['mal'])
                                      for u, v, d in self.graph[plane_type].edges(data=True)])
            for edge in self.graph[plane_type].edges:
                (u, v) = edge
                get_mal_edge_object = route_malfunction.get((u, v))
                total_malfunctions_in_episode += get_mal_edge_object.num_malfunctions

        return total_malfunctions_in_episode

    def get_available_routes(self, source: Airport, plane_type: PlaneType) -> Collection:
        """
        Gets all available routes from an Airport Node to ones it can directly travel to.

        :parameter source: Source Airport
        :parameter plane_model: Airplane Model
        :return: `frozenset` : a frozenset that contains what routes are disabled/malfunctioning

        """
        routes = self.graph[plane_type].adj[source.id]
        return [dest for dest, d in routes.items() if not d['mal'].in_malfunction]

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
