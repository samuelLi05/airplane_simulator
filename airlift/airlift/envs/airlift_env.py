import functools
import sys
from collections import namedtuple
from datetime import datetime
from enum import Enum
from math import floor
from statistics import mean
from typing import List, Dict, Tuple, NamedTuple, Optional, Collection
import pickle
import networkx as nx
from PIL.Image import Image
from gym import logger, Space
from gym.utils import seeding
from networkx import DiGraph
from pettingzoo.utils.env_logger import EnvLogger

from airlift.envs.plane_types import PlaneTypeID
from airlift.envs.airport import NOAIRPORT_ID, AirportID
from airlift.envs.cargo import CargoID, Cargo
from airlift.envs.renderer import EnvRenderer, FlatRenderer, default_height_in_pixels
from pettingzoo import ParallelEnv
from airlift.envs.agents import EnvAgent, AgentID, PlaneState
from gym.spaces import Discrete, MultiBinary
import gym
import airlift.envs.spaces as airliftspaces
from airlift.envs.route_map import RouteMap

# This should be included in PettingZoo in a future release.
# See https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py
from airlift.envs.generators.world_generators import WorldGenerator
from airlift.utils.definitions import TEST_MODE
from airlift.utils.seeds import generate_seed

"""Environmnet information for a given epsiode"""
EnvInfo = namedtuple('EnvInfo',
                     ['number_of_agents',
                      'number_of_airports',
                      'number_of_initial_cargo',
                      'max_cargo_per_episode',
                      'avg_working_capacity',
                      'avg_processing_time',
                      'soft_deadline_multiplier',
                      'hard_deadline_multiplier',
                      'min_degree',
                      'max_degree',
                      'avg_degree',
                      'malfunction_rate',
                      'malfunction_max_steps',
                      'malfunction_min_steps',
                      'max_soft_deadline',
                      'max_hard_deadline',
                      'REWARD_CARGO_MISSED_PENALTY',
                      'REWARD_CARGO_LATE_PENALTY',
                      'REWARD_MOVEMENT_PENALTY'
                      ]
                     )

"""Metrics regarding an episode"""
Metrics = namedtuple('Metrics',
                     ['total_cost',
                      'total_scaled_cost',
                      'average_cost_per_plane',
                      'total_lateness',
                      'total_scaled_lateness',
                      'average_lateness_per_plane',
                      'total_steps',
                      'average_steps',
                      'total_waiting_steps',
                      'total_waiting_to_process_steps',
                      'total_waiting_for_route_steps',
                      'max_seconds_to_complete',
                      'total_malfunctions',
                      'missed_deliveries',
                      'total_rewards_for_all_agents',
                      'average_rewards_for_all_agents',
                      'score',
                      'total_cargo_generated',
                      'dynamic_cargo_generated'
                      ])

NUM_PRIORITIES = 10


class PlaneTypeObservation(NamedTuple):
    """Airplane type info for Observation. This is part of the state space"""
    id: int
    max_weight: float


class CargoObservation(NamedTuple):
    """Cargo info for Observation. This is part of the state space"""
    id: CargoID
    location: AirportID
    destination: AirportID
    weight: float
    earliest_pickup_time: int
    is_available: int
    soft_deadline: int
    hard_deadline: int


class ScenarioObservation(NamedTuple):
    processing_time: int


class AirliftEnv(ParallelEnv):
    """Controls all aspects of the simulation/environment.
    The primary interface is the step method, which provides observations/states to the agents and steps the environment
    to the next state given agent actions"""

    REWARD_CARGO_MISSED_PENALTY = 10
    REWARD_CARGO_LATE_PENALTY = 1
    REWARD_MOVEMENT_PENALTY = 0.01

    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 60}

    def __init__(self,
                 world_generator=None,
                 pickuponly=False,
                 verbose=False,
                 renderer=FlatRenderer()):

        super().__init__()
        self.renderer: EnvRenderer = None
        self.world_generator: WorldGenerator = world_generator

        self.total_steps_in_episode = 0

        self.routemap: RouteMap = None
        self.cargo_task_generator = None

        # self._max_episode_steps: Optional[int] = None
        self._elapsed_steps = 0

        self.obs_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}

        self._agents: Dict[AgentID, EnvAgent] = {}

        self.num_resets = 0

        self.total_rewards = {}

        self.cargo_by_id: Dict[CargoID, Cargo] = None

        self.renderer = renderer

        # For petting zoo
        self.possible_agents: List[AgentID] = ['a_' + str(i) for i in range(world_generator.num_agents)]
        self.agents: List[AgentID] = None

        self.pickuponly = pickuponly

        # Not required for ParallelEnv, but including it for convenience
        self.rewards = {}
        self.dones: Dict[AgentID, bool] = None
        self.info = None
        self.next_actions = None

        self._np_random = None
        self._np_random_action = None

        self._obs = None
        self._state = None
        self._state_graph_cache = {}

        self.max_cycles = None

        self.verbose = verbose

    def reset(self, seed=None) -> Dict:
        """
        Resets the environment and generates a new random realization. If called without a seed, a new realization is generated.

        :parameter seed: Environment seed
        :return: A dictionary containing the initial observations for all agents. Individual observations of agents can be accessed using the 'a_0...n' keys.

        """
        self.agents = list(
            self.possible_agents)  # Make a copy of the list so that we don't affect possible_agents when we remove done agents

        # If a new seed is passed in, re-seed the environment and world generator.
        # If environment has not been reset yet, also do the seeding regardless.
        if seed is not None or self._np_random is None:
            self._np_random, seed = seeding.np_random(seed)
            self._np_random_action, seed = seeding.np_random(seed + 3)

            self.world_generator.seed(seed=generate_seed(self._np_random))

            # Seed the spaces
            self.state_space().seed(seed=generate_seed(self._np_random))
            for ospace in self.observation_spaces.values():
                ospace.seed(seed=generate_seed(self._np_random))
            for aspace in self.action_spaces.values():
                aspace.seed(seed=generate_seed(self._np_random))

        self.routemap, airplanes, cargo = self.world_generator.generate()
        self.renderer.reset(self.routemap, airplanes)
        self.cargo_by_id = {}
        cargo_sorted = sorted(cargo, key=lambda c: c.id)
        self._add_cargo(cargo_sorted)

        assert len(airplanes) == len(self.possible_agents)
        self._agents = {i: a for i, a in zip(self.possible_agents, airplanes)}

        self.mark_cargo_delivered()  # If cargo has same source/delivery, we will mark it as delivered in the first step

        # Reset agents to initial states

        self.num_resets += 1
        self._elapsed_steps = 0

        # Not required for ParallelEnv, but including it for convenience
        self.clear_rewards_dict()
        self.total_rewards = {i_agent: 0 for i_agent in self.agents}
        self.dones = {a: False for a in self.agents}
        self.info = {a: {"warnings": []} for a in self.agents}
        self.next_actions = {a: ActionHelper.noop_action() for a in self.agents}
        self.next_actions = self.check_priority_for_none(self.next_actions)
        # Fill in the state graph which we return in the observation
        # We maintain this to avoid performance hit on building the dictionary
        self._state = {"route_map": {},
                       "plane_types": [PlaneTypeObservation(pt.id, pt.max_weight) for pt in self.routemap.plane_types],
                       "agents": {a: {} for a in self.agents}}

        for plane in self.routemap.plane_types:
            self._state["route_map"][plane.id] = nx.DiGraph(self.routemap.graph[plane])

            # Remove MalfunctionHandler class
            nx.set_edge_attributes(self._state["route_map"][plane.id], None, 'mal')

            # Clear out the airport object ref in the nodes - we don't want to pass this in the observation
            # Replace it with the airport id
            for node in self._state["route_map"][plane.id].nodes:
                nx.set_node_attributes(self._state["route_map"][plane.id], {node: {'airport': node}})

        # Maintain a cache to quickly link edges in the routemap graph to the state graph
        for plane in self.routemap.plane_types:
            edges = []
            for u, v, data_routemap in self.routemap.graph[plane].edges(data=True):
                data_state = self._state["route_map"][plane.id][u][v]
                edges.append((u, v, data_routemap, data_state))

            self._state_graph_cache[plane.id] = edges

        self._obs = {a: {} for a in self.agents}
        self._update_state_and_obs(new_cargo=[])

        self.max_cycles = self.world_generator.max_cycles
        return self.observe()

    def clear_rewards_dict(self):
        """ Reset the rewards dictionary """
        self.rewards = {i_agent: 0 for i_agent in self.agents}

    def mark_cargo_delivered(self):
        for cargo in self.cargo:
            if not self.cargo_delivered(cargo):
                # If the cargo was delivered, did it miss the soft or hard deadline.
                if self._elapsed_steps > cargo.soft_deadline:
                    cargo.missed_softdeadline = True
                if self._elapsed_steps > cargo.hard_deadline:
                    cargo.missed_hardeadline = True

        for airport in self.routemap.airports:
            for cargo in airport.cargo:
                if airport == cargo.end_airport and cargo.delivery_time is None:
                    cargo.delivery_time = self._elapsed_steps
                    logger.info("Cargo delivered at: " + str(cargo.delivery_time) + " Cargo ID: " + str(cargo.id))

    def _add_cargo(self, cargo: Collection[Cargo]):
        for c in cargo:
            self.cargo_by_id[c.id] = c

    @property
    def cargo(self) -> Collection[Cargo]:
        return list(self.cargo_by_id.values())

    def step(self, actions):
        """
        Steps the environment base don the given actions
        and returns a new observation. Based on the new observation, the policy should create another set of actions for the next step.

        :parameter actions: Dictionary that contains the actions for all agents
        :return: obs, Dictionary that contains the observation for all agents. rewards,Dictionary that contains rewards.
            dones, Dictionary that indicates if an agent has completed a scenario. info, A dictionary containing a list of Warnings for each agent.

        """

        if actions is None:
            actions = {a: None for a in self.agents}

        actions = self.check_priority_for_none(actions)

        if TEST_MODE:
            action_valid = ActionHelper.are_actions_valid(actions, self.observe())[0]
            if not action_valid:
                logger.warn("Invalid action(s) given")

        self.clear_rewards_dict()
        self._elapsed_steps += 1

        # Not allowed to step further once done
        if all(self.dones.values()):
            EnvLogger.warn_step_after_done()
        # Check route malfunctions
        self.routemap.step()

        # Generate Dynamic Cargo
        new_cargo = self.world_generator.cargo_generator.generate_dynamic_orders(self._elapsed_steps, self.max_cycles)
        self._add_cargo(new_cargo)

        for i in self.agents:
            agent = self._agents[i]

            # If passed in action is none, continue using last action
            if actions[i] is None:
                action = self.next_actions[i]
            else:
                action = actions[i]

            if TEST_MODE and not self.action_space(i).contains(action):
                # action = ActionHelper.noop_action()
                EnvLogger.warn_action_out_of_bound(action, self.action_space(i), action)

            updated_action, self.info[i]["warnings"] = agent.step(action, self.cargo_by_id, self._elapsed_steps)
            if self.verbose:
                for warning in self.info[i]["warnings"]:
                    logger.info("Agent {0}: {1}".format(i, warning))

            self.next_actions[i] = updated_action

            # Reward is based on cost/time of flight path.
            if agent.state == PlaneState.MOVING:
                cost_per_step = float(round(
                    self.routemap.get_flight_cost(agent.previous_airport,
                                                  agent.destination_airport, agent.plane_type) / \
                    self.routemap.get_flight_time(agent.previous_airport,
                                                  agent.destination_airport, agent.plane_type), 5))
                self.rewards[i] -= self.REWARD_MOVEMENT_PENALTY * cost_per_step

        self.mark_cargo_delivered()

        # Rewards related to cargo
        for cargo in self.cargo:
            if self.cargo_missed(cargo):
                if self._elapsed_steps == cargo.hard_deadline + 1:
                    for i in self.agents:
                        # We incur a penalty the moment the delivery is missed
                        self.rewards[i] -= self.REWARD_CARGO_MISSED_PENALTY
            elif self._elapsed_steps > cargo.soft_deadline:
                for i in self.agents:
                    # We haven't missed the hard deadline, but we are past the soft deadline.
                    # Incur a penalty for each step the delivery is late
                    self.rewards[i] -= self.REWARD_CARGO_LATE_PENALTY

        # Calculate dones
        if self._elapsed_steps >= self.max_cycles:
            logger.info("Episode is done - maximum steps reached")
            self.dones = {a: True for a in self.agents}
        elif self.pickuponly:
            # A simplified goal - each plane only has to pickup a cargo and then it's done
            for i in self.agents:
                agent = self._agents[i]
                if agent.cargo:
                    # This plane has cargo - it's done
                    self.dones[i] = True
                    logger.info("Agent {} has picked up cargo {} and is done".format(i, list(agent.cargo)[0]))
        elif all(self.cargo_done(c) for c in
                 self.cargo) and not self.world_generator.cargo_generator.will_generate_more_cargo():
            logger.info(
                "Episode is done - all cargo has been delivered or missed, and no more dynamic cargo will be generated")
            self.dones = {a: True for a in self.agents}

        # FOR DEBUGGING ONLY, CHECKS FOR CARGO ACTUALLY BEING ON A PLANE
        if TEST_MODE:
            for cargo in self.cargo:
                cargo_is_somewhere = False
                for airport in self.routemap.airports:
                    if cargo in airport.cargo:
                        cargo_is_somewhere = True
                for i in self.agents:
                    agent = self._agents[i]
                    if cargo in agent.cargo or cargo in agent.cargo_being_loaded or cargo in agent.cargo_being_unloaded:
                        cargo_is_somewhere = True
            assert cargo_is_somewhere

        # Update total rewards from rewards for this step
        for k in self.total_rewards.keys():
            self.total_rewards[k] += self.rewards.get(k, 0)  # Default to 0 reward if agent is done

        self._update_state_and_obs(new_cargo)

        # PettingZoo interface says an agent should be removed from the agents list once it's done
        for i, d in self.dones.items():
            if d and i in self.agents:
                self.agents.remove(i)

        self.total_steps_in_episode += 1

        return self.observe(), \
               self.rewards, \
               self.dones, \
               self.info

    def check_priority_for_none(self, actions):
        for i in self.agents:
            agent = self._agents[i]

            if actions[i] is not None and actions[i]['priority'] is None:
                actions[i]['priority'] = agent.priority
        return actions

    def cargo_lateness(self, cargo):
        if self.cargo_missed(cargo):
            return 0
        elif self.cargo_delivered(cargo):
            return max(0, cargo.delivery_time - cargo.soft_deadline)
        else:
            return max(0, self._elapsed_steps - cargo.soft_deadline)

    def cargo_max_lateness(self, cargo):
        return cargo.hard_deadline - cargo.soft_deadline

    def cargo_delivered(self, cargo):
        return cargo.delivery_time is not None

    def cargo_missed(self, cargo):
        return not self.cargo_delivered(cargo) and self._elapsed_steps > cargo.hard_deadline

    def cargo_done(self, cargo):
        assert not (self.cargo_delivered(cargo) and self.cargo_missed(cargo))
        return self.cargo_delivered(cargo) or self.cargo_missed(cargo)

    def set_render_options(self,
                           width_in_pixels=None,
                           height_in_pixels=None,
                           show_routes=None):
        self.renderer.set_render_options(width_in_pixels,
                                         height_in_pixels,
                                         show_routes)

    def render(self, mode="human"):
        if self.num_resets == 0:
            EnvLogger.error_render_before_reset()

        if mode == "human":
            self.renderer.render_to_window()
        elif mode == "rgb_array":
            return self.renderer.render_to_rgb_array()
        elif mode == "video":
            self.renderer.render_to_video()

    def render_to_file(self, filename):
        self.renderer.render_to_file(filename)

    def render_to_image(self, scale=1.0) -> Image:
        return self.renderer.render_to_image(scale)

    def close(self):
        """
        This methods closes any renderer window.
        """
        if self.num_resets == 0:
            EnvLogger.warn_close_before_reset()

        if self.renderer is not None:
            self.renderer.close_window()

    @property
    def metrics(self):
        """
        Returns metrics collected by the environment.
        Note: some values may be incorrect if this is called before the episode is done.
        """
        # assert all(self.dones.values())

        total_cost = sum(item.flight_cost for item in self._agents.values())

        total_lateness = 0
        total_scaled_lateness = 0
        missed_deliveries = 0
        for cargo in self.cargo:
            total_lateness += self.cargo_lateness(cargo)
            total_scaled_lateness += self.cargo_lateness(cargo) / max(self.cargo_max_lateness(cargo), 1)
            if self.cargo_missed(cargo):
                missed_deliveries += 1

        dynamic_cargo_generated = self.world_generator.cargo_generator.current_cargo_count - self.world_generator.cargo_generator.num_initial_tasks
        total_cargo_generated = self.world_generator.cargo_generator.num_initial_tasks + dynamic_cargo_generated
        assert total_cargo_generated == len(self.cargo)

        total_scaled_cost = 0
        for pt in self.routemap.plane_types:
            G = self.routemap.graph[pt]
            cost = sum(a.flight_cost for a in self._agents.values() if a.plane_type == pt)
            cap = sum(a.max_loaded_weight for a in self._agents.values() if a.plane_type == pt)

            sccs = list(nx.strongly_connected_components(G))
            subGs = [G.subgraph(nodes).copy() for nodes in sccs]
            shortestpaths = [{s: {t: nx.shortest_path_length(G, s, t, weight='cost') for t in nodes} for s in nodes} for
                             nodes in sccs]
            eccentricities = [nx.eccentricity(subG, sp=sp) for subG, sp in zip(subGs, shortestpaths)]
            diameters = [nx.diameter(subG, e=e) for subG, e in zip(subGs, eccentricities)]
            total_diameter = sum(diameters)
            total_scaled_cost += (cost / total_diameter) * (cap / total_cargo_generated)

        metrics = Metrics(
            total_cost=total_cost,
            average_cost_per_plane=total_cost / self.world_generator.num_agents,
            total_lateness=total_lateness,
            total_scaled_lateness=total_scaled_lateness,
            average_lateness_per_plane=total_lateness / self.world_generator.num_agents,
            total_steps=self.total_steps_in_episode,
            average_steps=self.total_steps_in_episode / self.world_generator.num_agents,
            total_waiting_steps=sum(item.waiting_steps for item in self._agents.values()),
            total_waiting_to_process_steps=sum(item.waiting_to_process_steps for item in self._agents.values()),
            total_waiting_for_route_steps=sum(item.waiting_for_route_steps for item in self._agents.values()),
            max_seconds_to_complete=600 + 10 * (self.total_steps_in_episode - 1),
            total_scaled_cost=total_scaled_cost,
            total_malfunctions=self.routemap.get_total_malfunctions(),
            missed_deliveries=missed_deliveries,
            total_rewards_for_all_agents=sum(self.total_rewards.values()),
            average_rewards_for_all_agents=mean(self.total_rewards.values()),
            score=self.REWARD_CARGO_MISSED_PENALTY * missed_deliveries \
                  + self.REWARD_CARGO_LATE_PENALTY * total_scaled_lateness \
                  + self.REWARD_MOVEMENT_PENALTY * total_scaled_cost,
            total_cargo_generated=total_cargo_generated,
            dynamic_cargo_generated=dynamic_cargo_generated
        )
        return metrics

    @property
    def env_info(self):
        """
        Returns environment info.
        Should not be called before the environment is reset.
        """
        assert self.num_resets > 0

        min_degree = 0
        max_degree = 0
        mean_degree = 0
        for plane in self.routemap.plane_types:
            degree = [d[1] for d in self.routemap.graph[plane].degree]
            max_degree += max(degree)
            min_degree += min(degree)
            mean_degree += mean(degree)

        env_info = EnvInfo(
            number_of_agents=self.world_generator.num_agents,
            number_of_airports=len(self.routemap.airports),
            number_of_initial_cargo=self.world_generator.cargo_generator.num_initial_tasks,
            max_cargo_per_episode=self.world_generator.max_cargo_per_episode,
            avg_working_capacity=(sum(item.allowed_capacity for item in self.routemap.airports)) / len(
                self.routemap.airports),
            avg_processing_time=(sum(item.processing_time for item in self.routemap.airports)) / len(
                self.routemap.airports),
            soft_deadline_multiplier=self.world_generator.cargo_generator.soft_deadline_multiplier,
            hard_deadline_multiplier=self.world_generator.cargo_generator.hard_deadline_multiplier,
            min_degree=min_degree,
            max_degree=max_degree,
            avg_degree=mean_degree,
            malfunction_rate=self.routemap.poisson_dist.lambda_value,
            malfunction_max_steps=self.world_generator.route_generator.malfunction_generator.max_duration,
            malfunction_min_steps=self.world_generator.route_generator.malfunction_generator.min_duration,
            max_soft_deadline=max(c.soft_deadline for c in self.cargo),
            max_hard_deadline=max(c.hard_deadline for c in self.cargo),
            REWARD_CARGO_MISSED_PENALTY=self.REWARD_CARGO_MISSED_PENALTY,
            REWARD_CARGO_LATE_PENALTY=self.REWARD_CARGO_LATE_PENALTY,
            REWARD_MOVEMENT_PENALTY=self.REWARD_MOVEMENT_PENALTY
        )

        return env_info

    @property
    def _largest_cargo_id(self):
        # For some test cases, we don't generate cargo.
        # In these cases, we will default the value to 1, to avoid Discrete(0) in the spaces (since this is not allowed).
        return max(self.world_generator.max_cargo_per_episode, 1)

    def _agent_space(self):
        return gym.spaces.Dict({
            "state": gym.spaces.Discrete(max(s.value for s in PlaneState) + 1),
            "current_airport": gym.spaces.Discrete(self.world_generator.max_airports + 1),
            "cargo_onboard": airliftspaces.List(Discrete(self._largest_cargo_id),
                                                self.max_cargo_on_airplane),
            "current_weight": Discrete(10000),
            "plane_type": gym.spaces.Discrete(len(self.world_generator.plane_types)),
            "max_weight": Discrete(10000),
            "cargo_at_current_airport": airliftspaces.List(Discrete(self._largest_cargo_id),
                                                           self.max_cargo_on_airplane),
            "available_routes": airliftspaces.List(Discrete(self.world_generator.max_airports + 1),
                                                   self.world_generator.max_airports),
            "next_action": self.action_space(self.agents[0]),
            "destination": gym.spaces.Discrete(self.world_generator.max_airports + 1)
        })

    @functools.lru_cache(maxsize=None)  # Ensures that we always return the same space (not a copy)
    def state_space(self) -> Space:
        route_map = gym.spaces.Dict({})
        for plane in self.world_generator.plane_types:
            route_map[plane.id] = airliftspaces.DiGraph(self.world_generator.max_airports + 1,
                                                        ["working_capacity"],
                                                        ["cost", "time", "route_available", "mal", "expected_mal_steps"])

        assert min(s.value for s in PlaneState) >= 0  # Make sure the enum will fit into a Discrete space

        cargo_info_space = airliftspaces.NamedTuple(CargoObservation, {'id': Discrete(self._largest_cargo_id),
                                                                       'location': Discrete(
                                                                           self.world_generator.max_airports + 1),
                                                                       'destination': Discrete(
                                                                           self.world_generator.max_airports + 1),
                                                                       'weight': Discrete(10000),
                                                                       'earliest_pickup_time': Discrete(100000),
                                                                       'is_available': Discrete(2),
                                                                       'soft_deadline': Discrete(100000),
                                                                       'hard_deadline': Discrete(100000)
                                                                       })

        scenario_observation = airliftspaces.NamedTuple(ScenarioObservation, {'processing_time': Discrete(1000)})
        return gym.spaces.Dict({
            "route_map": route_map,
            "active_cargo": airliftspaces.List(cargo_info_space, self.world_generator.max_cargo_per_episode),
            "plane_types": airliftspaces.List(
                airliftspaces.NamedTuple(PlaneTypeObservation, {'id': Discrete(len(self.world_generator.plane_types)),
                                                                'max_weight': Discrete(10000)}),
                len(self.world_generator.plane_types)),
            "event_new_cargo": airliftspaces.List(cargo_info_space, self.world_generator.max_cargo_per_episode),
            "agents": gym.spaces.Dict({a: self._agent_space() for a in self.possible_agents}),
            "scenario_info": airliftspaces.List(scenario_observation, 1)
        })

    def state(self):
        """
        Returns the complete state of the environment.
        """
        if self.num_resets == 0:
            EnvLogger.error_state_before_reset()
        return self._state

    def _get_cargo_location_ids(self, cargo) -> Tuple[AirportID]:
        # Find which airport the cargo is and return that airport
        for airport in self.routemap.airports:
            if cargo in airport.cargo:
                return airport.id

        if TEST_MODE:
            # If we need to find the cargo on an airplane, we can use the following code
            # # The cargo is not at an airport. It must be on an airplane.
            # # Find which airplane the cargo is on and return that airplane.
            for id, airplane in self._agents.items():
                if cargo in airplane.cargo or cargo in airplane.cargo_being_loaded or cargo in airplane.cargo_being_unloaded:
                    return NOAIRPORT_ID

            assert False, "The cargo cannot be found - it is lost"

        return NOAIRPORT_ID

    @property
    def max_cargo_on_airplane(self):
        return self.world_generator.cargo_generator.max_cargo_per_episode

    @property
    @functools.lru_cache(maxsize=None)  # Ensures that we always return the same space (not a copy)
    def observation_spaces(self) -> Dict[AgentID, Space]:
        # Note: All agents have the same observation space
        assert min(s.value for s in PlaneState) >= 0  # Make sure the enum will fit into a Discrete space
        sp = self._agent_space()
        sp["globalstate"] = self.state_space()

        return {a: sp for a in self.agents}

    @functools.lru_cache(maxsize=None)  # Ensures that we always return the same space (not a copy)
    def observation_space(self, agent) -> Space:
        return self.observation_spaces[agent]

    #  Not required for ParallelEnv, but including it for convenience
    def observe(self, agent: AgentID = None):
        if self.num_resets == 0:
            EnvLogger.error_observe_before_reset()

        if agent is None:
            return self._obs
        else:
            return self._obs[agent]

    def _build_cargo_obs(self, cargo: Cargo):
        return CargoObservation(
            cargo.id,
            self._get_cargo_location_ids(cargo),
            cargo.end_airport.id,
            cargo.weight,
            cargo.earliest_pickup_time,
            cargo.is_available(self._elapsed_steps),
            cargo.soft_deadline,
            cargo.hard_deadline)

    def _build_scen_info_obs(self):
        # These parameters are currently static and do not change throughout a scenario
        # If max working_capacity or processing time ever become dynamic. This observation will have to be changed.
        return ScenarioObservation(self.world_generator.airport_generator.processing_time,
                                   )

    def _update_state_and_obs(self, new_cargo):
        for plane in self.routemap.plane_types:
            for u, v, data_routemap, data_state in self._state_graph_cache[plane.id]:
                data_state['mal'] = data_routemap['mal'].malfunction_down_counter
                data_state['route_available'] = not data_routemap['mal'].in_malfunction

        self._state["active_cargo"] = [self._build_cargo_obs(c) for c in self.cargo if not self.cargo_done(c)]
        self._state["event_new_cargo"] = [self._build_cargo_obs(c) for c in new_cargo]
        self._state["scenario_info"] = [self._build_scen_info_obs()]

        for agent in self.agents:
            agentobj = self._agents[agent]
            agentstate = self._state["agents"][agent]

            agentstate["cargo_onboard"] = [c.id for c in agentobj.cargo]
            agentstate["state"] = agentobj.state

            agentstate["plane_type"] = agentobj.plane_type.id
            agentstate["current_weight"] = sum(c.weight for c in agentobj.cargo)
            agentstate["max_weight"] = agentobj.plane_type.max_weight

            agentstate["available_routes"] = self.routemap.get_available_routes(agentobj.current_airport,
                                                                                agentobj.plane_type)

            agentstate['next_action'] = self.next_actions[agent]

            if agentobj.state == PlaneState.MOVING:
                # We can't set these to None - we need to put some value even though they're meaningless while airplane is in flight
                agentstate["current_airport"] = agentobj.current_airport.id
                agentstate["cargo_at_current_airport"] = []
                agentstate[
                    "destination"] = agentobj.destination_airport.id if agentobj.destination_airport is not None else NOAIRPORT_ID
            else:
                agentstate["current_airport"] = agentobj.current_airport.id
                agentstate["cargo_at_current_airport"] = [c.id for c in agentobj.current_airport.cargo if c.is_available(self._elapsed_steps)]
                agentstate[
                    "destination"] = agentobj.destination_airport.id if agentobj.destination_airport is not None else NOAIRPORT_ID

            # Update the observation using the agent's values in the state
            self._obs[agent].update(agentstate)
            self._obs[agent]["globalstate"] = self._state

    # if TEST_MODE:
    #   assert self.state_space().contains(self._state)
    #  for agent in self.agents:
    #     assert self.observation_space(agent).contains(self._obs[agent])

    @property
    @functools.lru_cache(maxsize=None)
    def action_spaces(self) -> Dict[AgentID, Space]:
        sp = gym.spaces.Dict({
            "priority": Discrete(NUM_PRIORITIES),
            "cargo_to_load": airliftspaces.List(Discrete(self._largest_cargo_id),
                                                self.max_cargo_on_airplane),
            "cargo_to_unload": airliftspaces.List(Discrete(self._largest_cargo_id),
                                                  self.max_cargo_on_airplane),
            "destination": gym.spaces.Discrete(self.world_generator.max_airports + 1)
        })

        return {a: sp for a in self.possible_agents}  # All agents should the same observation space

    @functools.lru_cache(maxsize=None)  # Ensures that we always return the same space (not a copy)
    def action_space(self, agent: AgentID) -> Space:
        return self.action_spaces[agent]

    def sample_valid_actions(self):
        return ActionHelper.sample_valid_actions(self.observe(), self._np_random_action)

    def save(self, filename):
        with open(filename, 'wb') as out_pickle:
            pickle.dump(self, out_pickle)

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as file:
            env = pickle.load(file)

        return env


# Helper functions for making actions
class ActionHelper:
    """
    Assists policies in making actions for loading, unloading, processing, No-Op or taking off. Also includes a
    sample valid action function that is utilized for the Random Agent
    """

    def __init__(self, np_random=seeding.np_random()[0]):
        self._np_random = np_random
        self._randcache = []

    # We implement our own random generators so that we can use a cache of random numbers.
    # This can be much faster than making individual calls to the random generator.

    # Generate a random number
    def _rand(self):
        if not self._randcache:
            self._randcache = list(self._np_random.random(size=1000))
        return self._randcache.pop()

    # Choose an element from seq uniformly at random
    def _choice(self, seq):
        r = self._rand()
        i = floor(r * (len(seq) - sys.float_info.epsilon))
        return seq[i]

    # Choose a subset of the cargo uniformly at random
    def _sample_cargo(self, cargo):
        # Flip a coin to decide whether to include each cargo item
        # See "List" sample code in spaces.py for more info - this implementation should be equivalent
        if cargo is not None:
            cargo.sort()
        val = [c for c in cargo if self._rand() > 0.5]
        return val

    def sample_valid_actions(self, observation):
        actions = {}
        for a in observation:
            obs = observation[a]

            actions[a] = {"priority": 1,
                          "cargo_to_load": self._sample_cargo(obs["cargo_at_current_airport"]),
                          "cargo_to_unload": self._sample_cargo(obs["cargo_onboard"]),
                          "destination": self._choice([NOAIRPORT_ID] + list(obs["available_routes"]))}
        return actions

    @staticmethod
    def load_action(cargo_to_load) -> dict:
        """
        Loads Cargo by ID

        :parameter cargo_to_load: int,cargo ID to load
        :return: A dictionary containing all the actions the agent will take that includes the cargo to load.

        """
        return ActionHelper().process_action(cargo_to_load=cargo_to_load)

    @staticmethod
    def unload_action(cargo_to_unload) -> dict:
        """
        Unloads Cargo by ID

        :parameter cargo_to_unload: int, cargo ID to unload
        :return: A dictionary containing all the actions the agent will take that includes the cargo to unload.

        """
        return ActionHelper().process_action(cargo_to_unload=cargo_to_unload)

    @staticmethod
    def process_action(cargo_to_load=[], cargo_to_unload=[]) -> dict:
        """
        Processes an action

        :parameter cargo_to_load: A list containing the cargo IDs to load
        :parameter cargo_to_unload: A list containing cargo IDs to unload
        :return: A dictionary that contains all the cargo to load and unload.

        """
        if isinstance(cargo_to_load, int):
            cargo_to_load = [cargo_to_load]
        if isinstance(cargo_to_unload, int):
            cargo_to_unload = [cargo_to_unload]
        return {"priority": None, "cargo_to_load": cargo_to_load, "cargo_to_unload": cargo_to_unload,
                "destination": NOAIRPORT_ID}

    @staticmethod
    def takeoff_action(destination: int) -> dict:
        """
        Gives the next destination

        :parameter destination: int, destination ID airport
        :return: A dictionary containing the destination for the airplane to take off to.

        """
        return {"priority": None, "cargo_to_load": [], "cargo_to_unload": [], "destination": destination}

    @staticmethod
    def noop_action() -> dict:
        """
        No-Op Action, "Do nothing" (except refuel based on priority)

        :return: A dictionary where the values contained will make the airplane take no action.

        """
        return {"priority": None, "cargo_to_load": [], "cargo_to_unload": [], "destination": NOAIRPORT_ID}

    @staticmethod
    def is_noop_action(action) -> bool:
        """
        Checks if an action is a No-Op action.

        :parameter action: dictionary, contains actions of a single agent
        :return: Boolean, True if No-Op action.

        """
        return not action["cargo_to_load"] and \
               not action["cargo_to_unload"] and \
               action["destination"] == NOAIRPORT_ID

    @classmethod
    def are_actions_valid(cls, actions, obs):
        valid_dict = {}
        warnings_dict = {}
        for a in actions.keys():
            valid_dict[a], warnings_dict[a] = cls.is_action_valid(actions[a], obs[a])

        return all(valid_dict.values()), warnings_dict

    @staticmethod
    def is_action_valid(agent_action, agent_obs):
        state = agent_obs["globalstate"]

        warnings_list = []
        action_valid = True

        if agent_action is None:
            pass
        else:
            if agent_action["priority"] not in list(range(NUM_PRIORITIES)):
                warnings_list.append(ObservationHelper.ObservationWarnings.PRIORITY_OUT_OF_BOUNDS)

            if agent_action["destination"] not in (
                    [NOAIRPORT_ID] + agent_obs["available_routes"]):
                warnings_list.append(ObservationHelper.ObservationWarnings.ROUTE_UNAVAILABLE)

            max_airports = len(state["route_map"][agent_obs["plane_type"]].nodes)
            if agent_action["destination"] > max_airports:
                warnings_list.append(ObservationHelper.ObservationWarnings.AIRPORT_OUT_OF_BOUNDS)

            if any(c not in agent_obs['cargo_onboard'] for c in agent_action['cargo_to_unload']):
                warnings_list.append(ObservationHelper.ObservationWarnings.ATTEMPTING_TO_UNLOAD_CARGO_NOT_ONBOARD)

            if any(c not in agent_obs['cargo_at_current_airport'] for c in agent_action['cargo_to_load']):
                warnings_list.append(ObservationHelper.ObservationWarnings.ATTEMPTING_TO_LOAD_CARGO_NOT_AT_AIRPORT)

            weight_to_load = ObservationHelper.get_cargo_weight(state, agent_action["cargo_to_load"])
            weight_to_unload = ObservationHelper.get_cargo_weight(state, agent_action["cargo_to_unload"])
            if agent_obs["current_weight"] + weight_to_load - weight_to_unload > agent_obs["max_weight"]:
                warnings_list.append(ObservationHelper.ObservationWarnings.AGENT_MAX_WEIGHT_EXCEEDED)

        if warnings_list:
            action_valid = False
        return action_valid, warnings_list


# Helper functions for using the state and observations

class ObservationHelper:
    """Helper class for using the state and observation. All methods are static methods."""

    def __init__(self):
        self.obs_warnings = self.ObservationWarnings()

    # Indicates if the airplane is idle, i.e. it has no action assignment, and the plane is waiting or ready for takeoff.
    @staticmethod
    def needs_orders(airplane_obs):
        """
        Checks to see if an airplane is idle.

        :parameter airplane_obs: agent observation
        :return: Returns True if the Airplane is in the Waiting state, or ready for take off and does not have any actions assigned.

        """
        return ActionHelper.is_noop_action(airplane_obs["next_action"])

    @staticmethod
    def available_destinations(state, airplane_obs, plane_type: PlaneTypeID):

        """
        Returns available destination from an airport node.

        :parameter state: Airplane current state
        :parameter airplane_obs: Airplane Observation
        :parameter plane_type: Airplane Model by PlaneTypeID
        :return: Returns a list of all available destinations that the agent can travel to from its current node.

        """
        return [o for i, o in state["route_map"][plane_type].out_edges(airplane_obs["current_airport"])]

    @staticmethod
    def get_lowest_cost_path(state, airport1, airport2, plane_type: PlaneTypeID):

        """
        Gets the shortest path from airport1 to airport2 based on the plane model.

        :parameter state: Airplane current state
        :parameter airport1: From Airport
        :parameter airport2: To Airport
        :parameter plane_type: airplane model by PlaneTypeID
        :return: A list containing the shortest path from airport1 to airport2.

        """

        # Changed this to take into account if we pass in the view[a] directly.
        if not isinstance(state, DiGraph):
            # If the whole state is passed
            return nx.shortest_path(state["route_map"][plane_type], airport1, airport2, weight="cost")
        else:
            # If the view is passed
            return nx.shortest_path(state, airport1, airport2, weight="cost")

    @staticmethod
    def get_all_shorest_path(state, airport1, airport2, plane_type: PlaneTypeID):
        return nx.all_shortest_paths(state["route_map"][plane_type], airport1, airport2)

    @staticmethod
    def get_active_cargo_info(state, cargoid):
        """
        Gets current active cargo info. Active cargo is cargo that has not been assigned & delivered.

        :parameter state: Airplane current state
        :parameter cargoid: Cargo by ID
        :return: A list containing all the currently active cargo.

        """
        if cargoid is not None and isinstance(cargoid, list):
            cargo_infos = [ci for ci in state["active_cargo"] if ci.id in cargoid]
            if cargo_infos:
                return cargo_infos
        elif isinstance(cargoid, int):
            cargo_infos = [ci for ci in state["active_cargo"] if ci.id == cargoid]
            if cargo_infos:
                return cargo_infos[0]
        else:
            return None


    @classmethod
    def get_cargo_objects(cls, state, cargo_list):
        cargo_infos = [cls.get_active_cargo_info(state, c) for c in cargo_list]
        return cargo_infos

    @classmethod
    def get_cargo_weight(cls, state, cargo_list):
        # TODO: If an airplane has cargo onboard that becomes "inactive", we can't get the cargo weight from the observation.
        #       For now we just ignore that cargo. This could lead to incorrect results.
        cargo_infos = [cls.get_active_cargo_info(state, c) for c in cargo_list]
        return sum(ci.weight for ci in cargo_infos if ci is not None)

    @staticmethod
    def get_multidigraph(state) -> nx.MultiDiGraph():
        """
        Gets the routemap as a multigraph.

        :parameter state: Current Airplane state
        :return: A route map as a MultiDiGraph

        """
        return RouteMap.build_multigraph(state["route_map"])

    class ObservationWarnings(Enum):
        ROUTE_DISABLED = 1
        ROUTE_UNAVAILABLE = 2
        UNABLE_TO_LOAD_CARGO = 3
        ATTEMPTING_TO_UNLOAD_CARGO_NOT_ONBOARD = 4
        ATTEMPTING_TO_LOAD_CARGO_NOT_AT_AIRPORT = 5
        PRIORITY_OUT_OF_BOUNDS = 6
        AGENT_MAX_WEIGHT_EXCEEDED = 7
        AIRPORT_OUT_OF_BOUNDS = 8
