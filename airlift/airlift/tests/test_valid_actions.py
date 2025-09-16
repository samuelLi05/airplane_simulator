import numpy as np
from gym.utils import seeding
from airlift.envs.events.event_interval_generator import NoEventIntervalGen
from airlift.envs import ActionHelper, ObservationHelper as oh, StaticCargoGenerator, NOAIRPORT_ID, PlaneState
from airlift.tests.util import generate_environment
from gym import logger

logger.set_level(logger.WARN)


def test_legal_actions(render):
    seed = 546435
    np_random, seed = seeding.np_random(seed)
    num_airports = 5
    env = generate_environment(num_of_airports=num_airports, num_of_agents=1, processing_time=0,
                               malfunction_generator=NoEventIntervalGen(),
                               cargo_generator=StaticCargoGenerator(1, soft_deadline_multiplier=1000,
                                                                    hard_deadline_multiplier=2000, max_stagger_steps=0))
    obs = env.reset(seed)
    valid_destination(env, obs)
    valid_priority(env, obs)
    load_cargo(env)
    unload_cargo(env)


def valid_priority(env, obs):
    route = 4
    actions = {'a_0': {'priority': 5, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': route}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    env.step(actions)
    assert not action_legal

    # Check the valid priority [0,1]
    priority = [0, 1]
    for i in range(len(priority)):
        actions = {
            'a_0': {'priority': priority[i], 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': route}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        assert action_legal


def valid_destination(env, obs):
    # Not a fully connected graph, no malfunctions, check to see if we can get to node 4..
    available_routes = obs['a_0']['available_routes']
    route = np.random.choice(available_routes)
    actions = {'a_0': {'priority': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': route}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    env.step(actions)

    obs = env.observe()
    assert action_legal

    # Check to see if we can get to node > num_airports
    actions = {'a_0': {'priority': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': 100}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    obs = env.observe()
    assert not action_legal

    # Go through list of available routes.
    available_routes = obs['a_0']['available_routes']
    while available_routes:
        actions = {'a_0': {'priority': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(),
                           'destination': np.random.choice(available_routes)}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        available_routes.pop()
        assert action_legal

    # Go through a list of unavailable routes/out of bounds
    available_routes = [30, 60, 70, 80, 100]
    while available_routes:
        actions = {'a_0': {'priority': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(),
                           'destination': np.random.choice(available_routes)}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        available_routes.pop()
        assert not action_legal


def load_cargo(env):
    # Test real cargo we can load
    observation = env.observe()
    cargo_info = oh.get_active_cargo_info(env.state(), 0)
    actions = {a: None for a in env.agents}

    # Reach cargo location, load cargo
    while observation['a_0']['current_airport'] is not cargo_info.location or not observation['a_0']['cargo_onboard'] and observation['a_0']['state'] is not PlaneState.MOVING:
        for a in observation:
            obs = observation[a]
            actions[a] = {"priority": 0,
                          "cargo_to_load": [],
                          "cargo_to_unload": [],
                          "destination": NOAIRPORT_ID}

            if obs['current_airport'] is cargo_info.location:
                actions[a]["priority"] = 1
                actions[a]["cargo_to_load"].append(cargo_info.id)
                actions[a]['destination'] = NOAIRPORT_ID
                action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
                obs, rewards, dones, _ = env.step(actions)
                assert action_legal
            else:
                action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
                actions[a]['destination'] = cargo_info.location
                assert action_legal

        # Step ahead a bit to account for processing time, currently set to 0 but still need +1 step.
        actions = {a: None for a in env.agents}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
        obs, rewards, dones, _ = env.step(actions)
        assert action_legal
        assert obs['a_0']['cargo_onboard']

    # Test Cargo that does not exist
    for a in observation:
        obs = observation[a]
        actions[a] = {"priority": 0,
                      "cargo_to_load": [],
                      "cargo_to_unload": [],
                      "destination": NOAIRPORT_ID}

        actions[a]["priority"] = 1
        actions[a]["cargo_to_load"].append(5)
        actions[a]['destination'] = NOAIRPORT_ID
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
        obs, rewards, dones, _ = env.step(actions)
        assert not action_legal

    # Make sure cargo is onboard



# Not sure if this part of the test is working properly
def unload_cargo(env):
    observation = env.observe()
    cargo_info = oh.get_active_cargo_info(env.state(), 0)

    # Reach cargo destination, unload cargo
    while observation['a_0']['current_airport'] is not cargo_info.destination or (observation['a_0']['cargo_onboard'] and observation['a_0']['state'] is not PlaneState.MOVING):
        actions = {a: None for a in env.agents}
        for a in observation:
            obs = observation[a]
            actions[a] = {"priority": 1,
                          "cargo_to_load": [],
                          "cargo_to_unload": [],
                          "destination": NOAIRPORT_ID}

            if obs['current_airport'] is cargo_info.destination:
                actions[a]["cargo_to_unload"].append(cargo_info.Aid)
                action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
                observation, rewards, dones, _ = env.step(actions)
                assert action_legal
            else:
                action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
                actions[a]['destination'] = cargo_info.destination
                observation, rewards, dones, _ = env.step(actions)
                assert action_legal
    # Step ahead a bit to account for processing time, currently set to 0 but still need +1 step.
    actions = {a: None for a in env.agents}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
    assert action_legal
    obs, rewards, dones, _ = env.step(actions)

    # Try and unload some random cargo
    for a in observation:
        obs = observation[a]
        actions[a] = {"priority": 0,
                      "cargo_to_load": [],
                      "cargo_to_unload": [],
                      "destination": NOAIRPORT_ID}

        actions[a]["priority"] = 1
        actions[a]["cargo_to_unload"].append(5)
        actions[a]['destination'] = NOAIRPORT_ID
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, observation)
        obs, rewards, dones, _ = env.step(actions)
        assert not action_legal

    # Make sure cargo is no longer in airplane
    assert not obs['a_0']['cargo_onboard']
