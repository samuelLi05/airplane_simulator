from __future__ import annotations

import itertools
from typing import Set
from airlift.envs.world_map import Coordinate

# Avoiding circular imports by importing TYPE_CHECKING for typehints
from typing import TYPE_CHECKING

from airlift.utils.airplane_queue import AirplaneQueue

if TYPE_CHECKING:
    from airlift.envs.cargo import Cargo
    from airlift.envs.agents import EnvAgent

AirportID = int
NOAIRPORT_ID = 0


class Airport:
    """
    Airports are represented by nodes in the route map graph.
    """

    def __init__(self, pos, id, processing_time=1, working_capacity=10 ** 6):
        self.position: Coordinate = pos
        self.id: AirportID = id
        self.cargo: Set[Cargo] = set()
        self.allowed_capacity = working_capacity
        self.agents_processing = set()
        self.agents_waiting = AirplaneQueue()
        self.processing_time = processing_time
        self.counter = itertools.count()
        self.in_drop_off_area = False
        self.in_pick_up_area = False

    def add_cargo(self, cargo: Cargo):
        """
        Adds a cargo item to an airport.

        :parameter cargo: Cargo to be added
        """
        self.cargo.add(cargo)

    def remove_cargo(self, cargo: Cargo):
        """
        Removes cargo from the airport's cargo list

        :parameter cargo: Cargo to be removed

        """
        self.cargo.remove(cargo)

    def add_to_waiting_queue(self, agent: EnvAgent):
        count = next(self.counter)
        self.agents_waiting.add_to_waiting_queue(agent, count)

    def add_to_capacity(self, agent: EnvAgent) -> None:
        """
        Adds an agent to the processing queue of an airport if it has capacity available.

        :parameter agent: Adds an agent to Airport capacity for processing.

        """
        if len(self.agents_processing) < self.allowed_capacity:
            assert agent not in self.agents_processing
            self.agents_processing.add(agent)

    def remove_from_capacity(self, agent: EnvAgent) -> None:
        """
        Removes the agent from the processing queue.

        :parameter agent: Removes an agent from an airport capacity that has completed processing.

        """

        if self.agents_processing:
            assert agent.processing_time_left == 0
            self.agents_processing.remove(agent)
            self.agents_waiting.agent_complete(agent)
            assert agent not in self.agents_processing

            # If there are no more agents waiting, initialize a new counter
            if self.agents_waiting.is_added_agents_empty():
                self.counter = itertools.count()

    def airport_has_capacity(self) -> bool:
        """
        Checks to see if the current airport has capacity in order to start processing an agent.

        :return: Boolean, True if there is space available

        """
        return len(self.agents_processing) < self.allowed_capacity
