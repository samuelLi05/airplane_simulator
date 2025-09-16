import networkx as nx
import pytest

from airlift.envs.airlift_env import AirliftEnv, ActionHelper as ah

import time

# Fixture to generate and initialize environment for testing
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs import PlaneState

# To run (--render is optional):
# pytest unit_tests/test_agent.py --render
from airlift.tests.util import generate_environment


@pytest.fixture
def env():
    return generate_environment(num_of_agents=1, cargo_generator=StaticCargoGenerator(1, 100, 150, max_stagger_steps=0),)


@pytest.fixture
def env_with_malfunctions():
    return generate_environment(num_of_agents=1, cargo_generator=StaticCargoGenerator(1, 100, 150, max_stagger_steps=0), num_of_airports=3, malfunction_generator=EventIntervalGenerator(min_duration=1, max_duration=2))


def check_reward(env, a):
    agent = env._agents.get(a, None)

    # if env.observe(a)["state"] == PlaneState.MOVING:  # and env._agents[agent].last_state == PlaneState.MOVING:
    #     assert env.rewards[a] == -float(round(
    #             env.routemap.get_flight_cost(agent.previous_airport,
    #                                          agent.destination_airport, agent.plane_type) / \
    #             env.routemap.get_flight_time(agent.previous_airport,
    #                                          agent.destination_airport, agent.plane_type), 5))
    # else:
    #     assert env.rewards[a] == 0


# Assumes action doesn't involve movement
def do_single_agent_action(env, agent, action, interimstate, donestate, render):
    ah.is_action_valid(action, env.observe(agent))[0] # We don't assert this, since routes may be unavailable in some cases...
    env.step({agent: action})
    check_reward(env, agent)

    while env.observe(agent)["state"] != donestate and not env.dones[agent]:
        # assert env.observe(agent)["state"] == interimstate
        env.step({agent: None})
        check_reward(env, agent)

        if render:
            env.render()
            time.sleep(0.1)


def follow_path(env, agent, path, render):
    assert env.observe(agent)["current_airport"] == path[0]
    for nextairport in path[1:]:
        assert env.observe(agent)["state"] == PlaneState.READY_FOR_TAKEOFF
        do_single_agent_action(env, agent, ah.takeoff_action(nextairport), PlaneState.MOVING, PlaneState.WAITING,
                               render)
        assert env.observe(agent)["current_airport"] == nextairport


# Tests
def test_malfunctions(env_with_malfunctions, render):
    test_fullrun(env_with_malfunctions, render)

def test_saved_env(env, render, tmp_path):
    filename = "level_0.pkl"
    env.save(filename)
    env = AirliftEnv.load(filename)

    test_fullrun(env, render)


def test_fullrun(env, render):
    """Test action space"""

    obs = env.reset(seed=10)

    assert len(env.agents) == 1
    agent = env.agents[0]
    agentobj = env._agents[agent]
    list_of_active_tasks = env.state()["active_cargo"]
    assert len(list_of_active_tasks) == 1
    task = list(list_of_active_tasks)[0]

    # Plane should start ready to takeoff
    assert env.observe(agent)["state"] == PlaneState.READY_FOR_TAKEOFF
    # do_single_agent_action(env,
    #                        agent,
    #                        env.build_process_action(),
    #                        PlaneState.PROCESSING,
    #                        PlaneState.READY_FOR_TAKEOFF,
    #                        render)

    # env.step({agent.handle: (NO_OP, -1)})
    # assert not env.lastdone['__all__']

    # print(env.observation_space(0))
    # print(env.state_space())

    # Go pick up the cargo
    obs = env.observe(agent)
    path = nx.shortest_path(env.state()["route_map"][obs["plane_type"]], obs["current_airport"], task.location,
                            weight="cost")
    follow_path(env, agent, path, render)
    assert env.observe(agent)["current_airport"] == task.location

    # # Testing flattening spaces - this should really be done somewhere else, maybe a Jupyter notebook
    # os = env.observation_space(0)
    # test = gym.spaces.flatdim( os["cargo_at_current_airport"] )
    # test2 = gym.spaces.flatten(os["cargo_at_current_airport"], env.observe(0)["cargo_at_current_airport"])
    #
    # states = env.state_space()
    # test3 = gym.spaces.flatdim( states["route_map"] )
    # test4 = gym.spaces.flatten( states["route_map"], env.state()["route_map"] )
    #
    # test5 = gym.spaces.flatten(os, env.observe(0))
    # test6 = gym.spaces.flatten(states, env.state())

    assert env.observe(agent)["cargo_at_current_airport"] == [task.id]
    assert not env.observe(agent)["cargo_onboard"]
    do_single_agent_action(env, agent, ah.load_action(0), PlaneState.PROCESSING, PlaneState.READY_FOR_TAKEOFF,
                           render)  # TODO: Need to test how many cycles the plane is processing for - not sure that is correct
    assert not env.observe(agent)["cargo_at_current_airport"]
    assert env.observe(agent)["cargo_onboard"] == [task.id]

    # Deliver the cargo
    path = nx.shortest_path(env.state()["route_map"][obs["plane_type"]], env.observe(agent)["current_airport"], task.destination, weight="cost")
    follow_path(env, agent, path, render)

    assert not any(env.dones.values())

    assert not env.observe(agent)["cargo_at_current_airport"]

    do_single_agent_action(env, agent, ah.unload_action(0), PlaneState.PROCESSING, PlaneState.READY_FOR_TAKEOFF, render)
    assert env.observe(agent)["cargo_at_current_airport"] == [task.id]
    assert not env.observe(agent)["cargo_onboard"]

    assert all(env.dones.values())

    env.close()
