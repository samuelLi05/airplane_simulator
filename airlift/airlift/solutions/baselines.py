import random

import networkx as nx
from networkx import NetworkXNoPath
from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID, ObservationHelper
from airlift.solutions import Solution


# Random agent which chooses only valid actions
class RandomAgent(Solution):
    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)
        self._action_helper = ActionHelper(np_random=self._np_random)

    def policies(self, obs, dones, infos=None):
        return self._action_helper.sample_valid_actions(observation=obs)


class ShortestPath(Solution):
    def __init__(self):
        super().__init__()

        self.cargo_delivered = None
        self._full_delivery_paths = None
        self.multidigraph = None
        self.multi_view = None
        self.cargo_assignments = None
        self.path = None
        self.plane_graph = None
        self.view = None
        self.plane_types = []

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)

        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.view = {}
        self._full_delivery_paths = {}
        self.cargo_delivered = {a: [] for a in self.agents}

    def policies(self, obs, dones, infos=None):
        actions = {}
        state = self.get_state(obs)

        # Build views in-case of malfunctions
        self.build_multidigraph_view(state)
        self.build_agent_views(obs, state)

        # Create full delivery path for all active cargo
        self.get_initial_full_delivery_path(state)
        pending_cargo = self.get_active_cargo_from_state(state)

        # Create our cargo bins
        cargo_bins = self.separate_cargo_by_location_and_destination(pending_cargo)

        # Generate an initial plan of action
        self.get_initial_agent_actions(obs, state, dones, pending_cargo, cargo_bins)
        actions.update(self.plan(obs))

        # Use ActionHelper and ObservationHelper to update the action again in-case of non-valid entries.
        self.check_for_valid_actions(obs, state, actions)

        return actions

    def plan(self, obs):
        state = self.get_state(obs)
        actions = {a: None for a in self.agents}

        for a in self.agents:

            if oh.needs_orders(obs[a]):
                actions[a] = {"priority": 1,
                              "cargo_to_load": [],
                              "cargo_to_unload": [],
                              "destination": NOAIRPORT_ID}

                if self.path[a]:
                    next_destination = self.path[a][0]

                    if obs[a]["current_airport"] == next_destination:
                        self.path[a].pop(0)
                        if self.path[a]:
                            next_destination = self.path[a][0]

                    actions[a]["destination"] = next_destination
                ca = oh.get_active_cargo_info(state, self.cargo_assignments[a])
                if ca is not None:
                    if ca[0].id in obs[a]["cargo_onboard"]:
                        for cargo in ca:
                            if cargo.destination == obs[a]['current_airport'] or len(self.path[a]) == 0:
                                actions[a]["cargo_to_unload"].append(cargo.id)
                                # self.cargo_assignments[a].remove(cargo) <--Breaks things for some reason..
                                # Just setting it to None seems to fix it, but can't even set it to None outside the loop...
                                self.cargo_assignments[a] = None

                    elif ca[0].id in obs[a]['cargo_at_current_airport']:
                        assert ca[0].destination != obs[a]['current_airport']
                        for cargo in ca:
                            if cargo.id not in self.cargo_delivered[a]:
                                actions[a]["cargo_to_load"].append(cargo.id)
                                self.cargo_delivered[a].append(cargo.id)

        return actions

    def build_agent_views(self, obs, state):
        if not self.plane_types:
            for a in self.agents:
                if obs[a]['plane_type'] not in self.plane_types:
                    self.plane_types.append(obs[a]['plane_type'])

        # Add the subgraph view for each plane_type
        for plane_type in self.plane_types:
            self.view[plane_type] = nx.subgraph_view(state["route_map"][plane_type],
                                                     filter_edge=self.filter_edge)

    def build_multidigraph_view(self, state):
        self.multidigraph = oh.get_multidigraph(state)
        self.multi_view = nx.subgraph_view(self.multidigraph, filter_edge=self.filter_multi_graph_edge)

    def get_initial_full_delivery_path(self, state):
        assert all(c.location != c.destination for c in state["active_cargo"])
        for c in state["active_cargo"]:
            if c.location != NOAIRPORT_ID and c.id not in self._full_delivery_paths:
                try:
                    self._full_delivery_paths[c.id] = nx.shortest_path(self.multi_view, c.location, c.destination,
                                                                       weight="cost")[1:]
                except nx.NetworkXNoPath as e:
                    continue

    def get_active_cargo_from_state(self, state):
        pending_cargo = [c for c in state["active_cargo"] if
                         c.id not in self.cargo_assignments.values() and c.is_available == 1]
        return pending_cargo

    def separate_cargo_by_location_and_destination(self, cargo_list):

        cargo_bins = {}

        for cargo_item in cargo_list:
            location = cargo_item.location
            destination = cargo_item.destination
            key = (location, destination)

            if key not in cargo_bins:
                cargo_bins[key] = []
            if cargo_item.is_available:
                cargo_bins[key].append(cargo_item)

        return cargo_bins

    def select_cargo(self, cargo_bins):
        keys_list = list(cargo_bins.keys())
        random_key_from_list = tuple(self._np_random.choice(keys_list))
        cargo_info = cargo_bins[random_key_from_list]
        return random_key_from_list, cargo_info

    def get_initial_agent_actions(self, obs, state, dones, pending_cargo, cargo_bins):
        active_cargo_ids = [c.id for c in pending_cargo]
        for a in self.agents:
            plane_type = obs[a]['plane_type']
            self.plane_graph = state["route_map"][plane_type]

            # Create a copy of the cargo assignments and remove anything that shouldn't be there
            if self.cargo_assignments[a] is not None:
                cargo_copy = list(self.cargo_assignments[a])
                for cargo in cargo_copy:
                    if cargo not in active_cargo_ids:
                        self.cargo_assignments[a].remove(cargo)

            if dones[a]:
                continue

            if pending_cargo and self.cargo_assignments[a] is None:
                # Select cargo info and return the randomly selected key (loc, dest) pair.
                key, cargo_info = self.select_cargo(cargo_bins)
                if cargo_info[0].id not in self.cargo_delivered[a]:
                    if cargo_info[0].location != NOAIRPORT_ID:
                        if cargo_info[0].id in self._full_delivery_paths:
                            full_delivery_path = self._full_delivery_paths[cargo_info[0].id]
                        else:
                            try:
                                full_delivery_path = nx.shortest_path(self.multi_view, cargo_info[0].location,
                                                                      cargo_info[0].destination,
                                                                      weight="cost")  # [1:]
                            except NetworkXNoPath as e:
                                continue
                        try:
                            if full_delivery_path:
                                if not self.view[plane_type].has_edge(cargo_info[0].location, full_delivery_path[0]
                                                                      ):
                                    continue
                                path = oh.get_lowest_cost_path(self.view[plane_type], obs[a]["current_airport"],
                                                               cargo_info[0].location,
                                                               obs[a]["plane_type"])

                                while full_delivery_path and self.view[plane_type].has_edge(path[-1],
                                                                                            full_delivery_path[0],
                                                                                            ):
                                    path.append(full_delivery_path.pop(0))
                                self.path[a] = path

                                # Get the airplanes max carrying capacity and assign it cargo
                                max_airplane_weight = obs[a]['max_weight']
                                num_cargo_assigned = 0
                                assigned_cargo = []
                                cargo_info_copy = list(cargo_info)
                                for cargo in cargo_info_copy:
                                    assigned_cargo.append(cargo.id)
                                    pending_cargo.remove(cargo)
                                    num_cargo_assigned += 1
                                    cargo_info.remove(cargo)

                                    if num_cargo_assigned == max_airplane_weight:
                                     break

                                # Delete the key value if there are no more cargo in this bin, so we don't select it anymore
                                # during the next iteration
                                if not cargo_info:
                                    del cargo_bins[key]

                                self.cargo_assignments[a] = assigned_cargo

                                # If there are no more bins to assign, break out of the loop
                                if not cargo_bins:
                                    break

                        except NetworkXNoPath as e:
                            continue

    def create_path_and_update_action(self, obs, a, cargo_info, full_delivery_path, actions):
        plane_type = obs[a]['plane_type']

        # If we can't make any progress
        if not self.view[plane_type].has_edge(cargo_info[0].location, full_delivery_path[0]):
            return False

        path = oh.get_lowest_cost_path(self.view[plane_type], obs[a]["current_airport"],
                                       cargo_info[0].destination,
                                       obs[a]["plane_type"])

        while full_delivery_path and self.view[plane_type].has_edge(path[-1], full_delivery_path[0]):
            path.append(full_delivery_path.pop(0))

        self.path[a] = path
        actions[a].destination = self.path[a].pop()
        actions[a]['cargo_to_unload'].add(cargo_info.id)

    def get_full_delivery_path_by_cargo_info(self, obs, cargo_info):
        self._full_delivery_paths[cargo_info[0].id] = nx.shortest_path(self.multi_view,
                                                                       obs['current_airport'],
                                                                       cargo_info[0].destination,
                                                                       weight="cost")[1:]

        for cargo in cargo_info:
            if cargo is not None:
                self._full_delivery_paths[cargo.id] = self._full_delivery_paths[cargo_info[0].id]

        return self._full_delivery_paths[cargo_info[0].id]

    def check_for_valid_actions(self, obs, state, actions):
        for a in self.agents:
            valid = ActionHelper.is_action_valid(actions[a], obs[a])
            if not valid[0]:
                cargo_info = ObservationHelper.get_cargo_objects(state, obs[a]['cargo_onboard'])
                if cargo_info:
                    if cargo_info[0] is not None:
                        try:
                            full_delivery_path = self.get_full_delivery_path_by_cargo_info(obs[a], cargo_info)
                        except NetworkXNoPath as e:
                            continue

                        if full_delivery_path:
                            try:
                                if not self.create_path_and_update_action(obs, a, cargo_info, full_delivery_path,
                                                                          actions):
                                    continue

                            except NetworkXNoPath as e:
                                continue

    def filter_edge(self, u, v):
        """Filter DiGraph, used for Airplane Types graphs"""
        return self.plane_graph[u][v]["mal"] == 0

    # Filter a multidigraph

    def filter_multi_graph_edge(self, u, v, key):
        """Filter the MultiDiGraph (created from collection of DiGraphs)"""
        return self.multidigraph[u][v][key]['mal'] == 0

    # Check to see if the subgraph edges/nodes exist in the multigraph for our assertion
    def is_subgraph_of_multigraph(self, subgraph, multigraph):
        """Not utilized, but can be used to assert that the newly created views are subgraphs of the multi di graph"""
        if not set(subgraph.nodes).issubset(set(multigraph.nodes)):
            return False

        for u, v, data in subgraph.edges(data=True):
            if not multigraph.has_edge(u, v):
                return False
        return True
