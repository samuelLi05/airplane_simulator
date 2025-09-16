from enum import IntEnum
from typing import Tuple, List, Set, Dict, Collection
from gym import logger

from airlift.envs.plane_types import PlaneType
from airlift.envs.cargo import Cargo, CargoID
from airlift.envs.world_map import Coordinate
from airlift.envs.route_map import RouteMap
from airlift.envs.airport import NOAIRPORT_ID, Airport
from airlift.utils.definitions import TEST_MODE

AgentID = str


class PlaneState(IntEnum):
    """
    Enumeration that defines the states an agent can be in

        - :0: Waiting - airplane is waiting to process at an airport
        - :1: Processing - airplane is refueling and loading/unloading cargo
        - :2: Moving - Airplane is in flight to its destination
        - :3: Ready for takeoff - airplane is ready for takeoff once it is given a destination
    """

    # Docstring may seem redundant here, but its mainly for the apidocs
    WAITING = 0
    PROCESSING = 1
    MOVING = 2
    READY_FOR_TAKEOFF = 3


class EnvAgent:
    """
    Represents an individual agent (airplane) in an environment.
    """

    def __init__(self,
                 start_airport,
                 routemap=None,
                 plane_type=None,
                 max_loaded_weight=None,
                 malfunction_generator=None,
                 priority=1):

        self.flight_cost = 0
        self.flight_start_position: Coordinate = None
        self.destination_airport: Airport = None
        self.current_airport: Airport = start_airport
        self.routemap: RouteMap = routemap
        self.total_flight_time = None

        self.priority = priority

        self.state = PlaneState.READY_FOR_TAKEOFF

        self.cargo_being_loaded: Set[Cargo] = set()
        self.cargo_being_unloaded: Set[Cargo] = set()

        self.elapsed_flight_time = None

        self.processing_time_left = 0
        self.cargo: Set[Cargo] = set()
        self.waiting_steps = 0
        self.waiting_to_process_steps = 0
        self.waiting_for_route_steps = 0

        self.last_direction = None

        self.max_loaded_weight = int(max_loaded_weight)
        self.plane_type: PlaneType = plane_type

        self.previous_airport = None

        self.malfunction_generator = malfunction_generator

    def _handle_waiting(self, action: dict, cargo_by_id: Dict[CargoID, Cargo], elapsed_time,
                        warnings: List[str]) -> None:
        """
        Handles trnasitions from the waiting state

        Parameters
        ----------
        action - A dictionary containing 'process', 'cargo to load' cargo to unload' and destination
        cargo_by_id - Dictionary containing the Cargo and CargoID
        warnings - List of warnings issued by the environment
        """

        if action["priority"] != 0:
            self.waiting_to_process_steps += 1

            # If we haven't added it to the queue yet, add it.
            self.current_airport.add_to_waiting_queue(self)

            success, warnings = self.try_to_process([cargo_by_id[id] for id in action["cargo_to_load"]],
                                                    [cargo_by_id[id] for id in action["cargo_to_unload"]],
                                                    elapsed_time,
                                                    warnings)
            if success:
                action["cargo_to_load"] = []
                action["cargo_to_unload"] = []

    def _handle_processing(self) -> None:
        """
        Handles trnasitions from the processing state
        """
        if self.processing_time_left > 0:
            self.processing_time_left -= 1
        else:

            for c in self.cargo_being_unloaded:
                self.current_airport.add_cargo(c)
            self.cargo_being_unloaded = set()

            self.cargo = self.cargo.union(self.cargo_being_loaded)
            self.cargo_being_loaded = set()

            self.current_airport.remove_from_capacity(self)
            self.state = PlaneState.READY_FOR_TAKEOFF
            assert (self.current_cargo_weight <= self.max_loaded_weight)

    def _handle_moving(self) -> None:
        """
        Handles transitions from the moving state.
        """
        self.elapsed_flight_time += 1
        if self.elapsed_flight_time > self.total_flight_time:
            self.current_airport = self.destination_airport
            self.destination_airport = None
            self.state = PlaneState.WAITING
            self.elapsed_flight_time = None

    def _handle_ready_for_takeoff(self, action, cargo_by_id: Dict[CargoID, Cargo], elapsed_time,
                                  warnings: List[str]) -> None:
        """
        Handles transition when when ready for take off. Checks if routes are reachable from the current airport,
        checks for routes being made unavailable due to malfunctions, gets the flight time and flight cost associated
        with the edge being taken. After the agent starts moving the destination is set to the default value of NOAIRPORT_ID
        which is set to 0.

        :Parameters:
        ----------
        `action` : A dictionary of actions that contains 'process', 'cargo_to_load', 'cargo_to_unload' and destination
        `cargo_by_id` : Dictionary containing the Cargo and CargoID
        `warnings` : List of warnings issued by the environment. Ex: If an action is given to an unavailable route
        """
        if action["cargo_to_load"] or action["cargo_to_unload"]:
            # The agent has requested a cargo change before taking off - go back to processing state

            # If we haven't added it to the queue yet, add it.
            self.current_airport.add_to_waiting_queue(self)
            success, warnings = self.try_to_process([cargo_by_id[id] for id in action["cargo_to_load"]],
                                                    [cargo_by_id[id] for id in action["cargo_to_unload"]],
                                                    elapsed_time,
                                                    warnings)
            if success:
                action["cargo_to_load"] = []
                action["cargo_to_unload"] = []

        elif action["destination"] != NOAIRPORT_ID:
            # We are ready for flight and a destination has been set

            new_destination = self.routemap.airports_by_id[action["destination"]]

            if not self.routemap.reachable(self.current_airport, self.routemap.airports_by_id[action["destination"]],
                                           self.plane_type):
                warnings.append("The destination airport is not reachable from here!")
                action["destination"] = NOAIRPORT_ID
            elif action["destination"] not in self.routemap.get_available_routes(self.current_airport,
                                                                                 self.plane_type):
                warnings.append("ROUTE FROM: " + str(self.current_airport.id) + " TO: " + str(new_destination.id) + \
                                " is currently unavailable. Change destination or wait for it to become available! The route will be unavailable for "
                                + str(
                    self.routemap.get_malfunction_time(self.current_airport.id, new_destination.id,
                                                       self.plane_type)) + " steps!")
                self.waiting_for_route_steps += 1
            else:
                self.elapsed_flight_time = 0

                self.destination_airport = new_destination

                flight_time = self.routemap.get_flight_time(self.current_airport,
                                                            new_destination, self.plane_type)

                self.flight_cost += self.routemap.get_flight_cost(self.current_airport,
                                                                  new_destination, self.plane_type)

                self.flight_start_position = self.current_airport.position

                self.state = PlaneState.MOVING

                self.total_flight_time = flight_time
                self.previous_airport = self.current_airport

                action["destination"] = NOAIRPORT_ID

    @staticmethod
    def _check_cargo(cargo_ids: Collection[CargoID], cargo_by_id: Dict[CargoID, Cargo], warnings: List[str]):
        """
        Filters out invalid cargo from the given cargo collection.

        :parameter cargo_ids: A list of CargoIDs
        :parameter cargo_by_id: Dictionary containing all the cargo and IDs
        :parameter warnings: A list containing any warnings. In this case if a particular cargo ID does not exist.

        """
        valid_ids = []
        for id in cargo_ids:
            if id not in cargo_by_id:
                warnings.append("Cargo {} does not exist".format(id))
            else:
                valid_ids.append(id)

        return valid_ids

    def step(self, action, cargo_by_id: Dict[CargoID, Cargo], elapsed_time) -> Tuple[dict, List[str]]:
        """
        Updates the agent's state.

        :parameter action: A dictionary of actions that contains 'process', 'cargo_to_load', 'cargo_to_unload' and 'destination'
        :parameter cargo_by_id: Dictionary containing the Cargo and CargoID

        :return: Tuple containing updated_action and warnings. The updated action contains any updated events to the action dict that occurred during the time step. The warnings list contains any issues with the current actions given.

        """

        self.last_state = self.state
        warnings: List[str] = []
        updated_action = action.copy()
        updated_action["cargo_to_load"] = self._check_cargo(updated_action["cargo_to_load"], cargo_by_id, warnings)
        updated_action["cargo_to_unload"] = self._check_cargo(updated_action["cargo_to_unload"], cargo_by_id, warnings)

        # Note: While the agent is queued up and waiting for processing, changing the priority will not have an effect on its position in the queue.
        #       The priority change will only take effect the next time the agent enters the processing queue.
        self.old_priority = self.priority
        if action['priority'] is not None:
            self.priority = action['priority']


        if self.state == PlaneState.READY_FOR_TAKEOFF:
            self._handle_ready_for_takeoff(updated_action, cargo_by_id, elapsed_time, warnings)
        elif self.state == PlaneState.MOVING:
            self._handle_moving()
        elif self.state == PlaneState.PROCESSING:
            self._handle_processing()
        elif self.state == PlaneState.WAITING:
            self.waiting_steps += 1
            self._handle_waiting(updated_action, cargo_by_id, elapsed_time, warnings)

        return updated_action, warnings

    def try_to_process(self,
                       cargo_to_load: Collection[Cargo],
                       cargo_to_unload: Collection[Cargo],
                       elapsed_time,
                       warnings: List[str]) -> bool:
        """
        Checks to see if the current airport has capacity. If there is capacity the agent will go into
        the PROCESSING state and the processing timer will be updated. The agent is added to the airports capacity.

        :parameter cargo_to_load: A set that contains cargo to load
        :parameter cargo_to_unload: A set that contains cargo to unload
        :return: Boolean, if successfully load or unloaded cargo

        """

        # If the agent is in the queue and the priority was updated.
        if self.priority != self.old_priority:
            # Check to see if the agent is in the queue and get its count - assume it's there (the agent should be in the queue before calling try_to_process)
            check_agent_in_queue, old_count = self.current_airport.agents_waiting.is_agent_in_queue(self)
            assert check_agent_in_queue

            # Update the queue
            airport_counter = self.current_airport.counter
            self.current_airport.agents_waiting.update_priority(self.old_priority, self.priority, old_count, airport_counter, self)

        success = False
        next_in_queue = self.current_airport.agents_waiting.peek_at_next()
        if self.current_airport.airport_has_capacity() and self == next_in_queue:
            self.unload_cargo(cargo_to_unload, warnings)

            # Test for test_loading_unload.py
            if TEST_MODE and cargo_to_unload and cargo_to_load:
                # Assert we unloaded all the cargo we needed to before loading
                assert cargo_to_unload not in list(self.cargo)

            self.load_cargo(cargo_to_load, elapsed_time, warnings)
            self.processing_time_left = self.current_airport.processing_time
            self.current_airport.add_to_capacity(self)
            self.state = PlaneState.PROCESSING

            # Assert successful removal
            queue_size_before = self.current_airport.agents_waiting.qsize()
            self.current_airport.agents_waiting.get()
            assert queue_size_before > self.current_airport.agents_waiting.qsize()

            success = True

        # Utilize this warnings list
        elif not self.current_airport.airport_has_capacity():
            self.state = PlaneState.WAITING
            warnings.append("Airport does not have capacity to process!")

        elif self != next_in_queue:
            self.state = PlaneState.WAITING
            warnings.append("The agent is not next in queue!")

        elif not self.current_airport.airport_has_capacity() and self != next_in_queue:
            self.state = PlaneState.WAITING
            warnings.append("There is no capacity to process and the airplane is not next in queue!")

        return success, warnings

    def load_cargo(self, cargo_to_load: Collection[Cargo], elapsed_steps, warnings: List[str]):
        """
        Checks to make sure airplane can load cargo and loads the cargo. Also checks to ensure that cargo is assigned to
        that airplane

        :parameter cargo_to_load: A list that contains the Cargo to load
        :parameter warnings: List of warnings issued by the environment. Ex: If an action is given to an unavailable route

        """
        total_loaded = 0
        for cargo in cargo_to_load:
            if cargo not in self.current_airport.cargo:
                warnings.append(
                    "Unable to load Cargo ID: " + str(cargo.id) + " is not at airport")
            elif cargo.earliest_pickup_time > elapsed_steps:
                time_rem = cargo.earliest_pickup_time - elapsed_steps
                warnings.append(
                    "Unable to load Cargo ID: " + str(cargo.id) +
                    ". The cargo is not ready yet and will be ready in " + str(time_rem) + " steps!")
            else:
                if (cargo.weight + total_loaded + self.current_cargo_weight) <= self.max_loaded_weight:
                    # Increment total loaded weight for this iteration
                    total_loaded += cargo.weight
                    self.current_airport.remove_cargo(cargo)
                    self.cargo_being_loaded.add(cargo)

                else:
                    warnings.append(
                        "Unable to load Cargo ID: " + str(cargo.id) + " due to exceeding allowed airplane weight limit")

    def unload_cargo(self, cargo_to_unload: Collection[Cargo], warnings: List[str]):

        """
        Does a check to ensure the correct cargo is being unloaded and removes the cargo from the airplane.

        :parameter cargo_to_unload: A list that contains the Cargo to unload
        :parameter warnings: List of warnings issued by the environment. Ex: If an action is given to an unavailable route

        """
        for cargo in cargo_to_unload:
            if cargo not in self.cargo:
                warnings.append("Unable to unload cargo, ID: {}, AIRPORT: {}" 
                            .format(cargo.id, self.current_airport.id))
        for cargo in self.cargo.copy():
            cargoInUnloadList = cargo in cargo_to_unload
            if cargo.missed_hardeadline or cargoInUnloadList:
                self.cargo.remove(cargo)
                self.cargo_being_unloaded.add(cargo)
                logger.debug("Airplane Type: " + str(self.plane_type.id) + " delivered cargo!")


    @property
    def current_cargo_weight(self) -> float:

        """
        Gets the total current cargo weight on an airplane

        :return: current cargo weight on an airplane

        """

        weight = sum(c.weight for c in self.cargo)
        return weight

    # Needed for using PriorityQueue, https://docs.python.org/3/library/queue.html
    # https://docs.python.org/3/library/heapq.html#module-heapq, O(log n)
    def __lt__(self, another_agent):
        if not isinstance(another_agent, EnvAgent):
            raise TypeError('Can only compare two EnvAgents')
        if self.priority > another_agent.priority:
            return True
        else:
            return False

    def __gt__(self, another_agent):
        if not isinstance(another_agent, EnvAgent):
            raise TypeError("Can only compare two EnvAgents")
        if self.priority < another_agent.priority:
            return True
        else:
            return False

    def __ge__(self, another_agent):
        if not isinstance(another_agent, EnvAgent):
            raise TypeError('Can only compare two EnvAgents')
        if self.priority <= another_agent.priority:
            return True
        else:
            return False

    def __le__(self, another_agent):
        if not isinstance(another_agent, EnvAgent):
            raise TypeError('Can only compare two EnvAgents')
        if self.priority >= another_agent.priority:
            return True
        else:
            return False
