from typing import List

import networkx as nx
from gym.utils import seeding
from gym import logger

from airlift.envs.airport import AirportID
from airlift.envs.agents import EnvAgent
from airlift.envs.route_map import RouteMap
from typing import NamedTuple


class AirplaneInfo(NamedTuple):
    start: AirportID
    capacity: int = 1


class AirplaneGenerator:
    """
    Generates airplanes in the environment. Places the agents at their starting positions as well as what their
    limitations are.
    """

    def __init__(self, num_of_agents: int):
        self.num_agents = num_of_agents
        self._np_random = None

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate(self, routemap: RouteMap) -> List[EnvAgent]:
        """
        Generates an EnvAgent at a start location for each plane model as defined in the plane_types.py file. The function
        ensures that there is at a minimum of 2 or more connected components at an airport before placing the airplane at 
        that location. This in turn makes sure that the agent is not stuck on one single airport. The dictionary keys are 
        accessible using the plane model: Ex: 0, 1..

        :parameter routemap: A Dictionary of DiGraphs that that contains all the routes specific plane models are able to traverse
        :return: `airplanes`, A list that contains all the EnvAgents

        """
        # First, place an airplane in each connected component (this ensures feasibility)
        airplanes = []
        for plane_type in routemap.plane_types:
            for connected_component_nodes in nx.strongly_connected_components(routemap.graph[plane_type]):
                if len(connected_component_nodes) > 1:  # Ignore airports that don't have routes
                    agent_start = self._np_random.choice(sorted(list(connected_component_nodes)))

                    airplanes.append(EnvAgent(start_airport=routemap.airports_by_id[agent_start],
                                              routemap=routemap,
                                              plane_type=plane_type,
                                              max_loaded_weight=plane_type.max_weight))
                if len(airplanes) >= self.num_agents:
                    break
            if len(airplanes) >= self.num_agents:
                break

        # Next, place the remaining airplanes uniformly at random among their corresponding graphs
        while len(airplanes) < self.num_agents:
            plane_type = self._np_random.choice(routemap.plane_types)

            # Don't place airplanes in airports where they are stuck
            connected_airports = [node for node, degree in routemap.graph[plane_type].degree() if degree > 0]
            if connected_airports:
                agent_start = self._np_random.choice(connected_airports)

                airplanes.append(EnvAgent(start_airport=routemap.airports_by_id[agent_start],
                                          routemap=routemap,
                                          plane_type=plane_type,
                                          max_loaded_weight=plane_type.max_weight))

            # We will let the environment assign agent ID's

        self._check_airplanes_valid(routemap, airplanes)

        return airplanes

    def _check_airplanes_valid(self, routemap: RouteMap, airplanes: List[EnvAgent]) -> None:
        """
        Makes sure every connected component of the graph has at least one airplane

        :parameter routemap: A dictionary that contains all the DiGraphs for each specific airplane model.
        :parameter airplanes: A list of EnvAgents

        """
        assert len(airplanes) == self.num_agents
        airplane_nodes = {p.current_airport.id for p in airplanes}
        for plane_type in routemap.plane_types:
            for connected_component_nodes in nx.strongly_connected_components(routemap.graph[plane_type]):
                if len(connected_component_nodes) > 1:
                    # Is there a plane whose current location is in the connected component?
                    if not connected_component_nodes.intersection(airplane_nodes):
                        logger.warn(
                            "A connected component does not have any airplanes of type {}".format(plane_type.id))