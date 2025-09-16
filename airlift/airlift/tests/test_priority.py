import itertools
import random

import numpy as np
from airlift.envs.agents import EnvAgent, PlaneState
from gym.utils import seeding
from airlift.envs.events.event_interval_generator import NoEventIntervalGen
from airlift.envs import ActionHelper, ObservationHelper as oh, HardcodedCargoGenerator, NOAIRPORT_ID, CargoInfo, \
    StaticCargoGenerator
from airlift.tests.util import generate_environment
from gym import logger

from airlift.utils.airplane_queue import AirplaneQueue

logger.set_level(logger.WARN)


def test_priority_actions(render):
    seed = 546435
    np_random, seed = seeding.np_random(seed)
    num_airports = 3
    cargo_info = [CargoInfo(id=0, source_airport_id=3, end_airport_id=1),
                  CargoInfo(id=1, source_airport_id=3, end_airport_id=1),
                  CargoInfo(id=2, source_airport_id=3, end_airport_id=1)]
    env = generate_environment(num_of_airports=num_airports, num_of_agents=3, processing_time=0,
                               malfunction_generator=NoEventIntervalGen(),
                               cargo_generator=HardcodedCargoGenerator(cargo_info))
    obs = env.reset(seed)
    agent_actions(env, obs)


def test_agent_priority():
    seed = 42
    random.seed(seed)
    counter = itertools.count()
    pq = AirplaneQueue()
    # Add an initial queue to the pq
    for i in range(1000):
        priority = random.randint(1, 1000)
        agent = EnvAgent(start_airport=1, max_loaded_weight=10, priority=priority)
        pq.put((priority, next(counter), agent))

    added_agents = 0
    while not pq.empty():

        agent = pq.get()
        next_agent = pq.peek_at_next()

        if not pq.empty():
            assert agent >= next_agent

        # Add agents to the queue during execution
        if added_agents <= 100:
            added_agents += 1
            prob = random.uniform(0, 1)
            if prob > .5:
                priority = random.randint(1, 100)
                agent = EnvAgent(start_airport=1, max_loaded_weight=10, priority=priority)
                count = next(counter)
                pq.put((priority, count, agent))

                # For any added agent, lets randomly update it
                new_priority = random.randint(1, 100)

                # Update the priority
                pq.update_priority(priority, new_priority, count, counter, agent)


def test_agent_order():
    random.seed(42)
    counter = itertools.count()
    pq = AirplaneQueue()

    priority1_list = []
    for i in range(100):
        priority1_list.append(EnvAgent(start_airport=1, max_loaded_weight=10, priority=1))

    priority2_list = []
    for i in range(100):
        priority2_list.append(EnvAgent(start_airport=1, max_loaded_weight=10, priority=2))

    random.shuffle(priority1_list)
    random.shuffle(priority2_list)
    for a in priority2_list:
        pq.add_to_waiting_queue(a, next(counter))
    for a in priority1_list:
        pq.add_to_waiting_queue(a, next(counter))

    # This is the order that the agents should appear in the priority queue
    queue_order = priority1_list + priority2_list

    # Go through each queue entry and make sure it is in the order we added it (and according to priority)
    while not pq.empty():
        agent = pq.get()

        agent2 = queue_order.pop(0)
        assert agent == agent2

def test_agent_step_priority():
    # This test just keeps the agents where they are and repeatedly loads/unloads a cargo item. The idea is to keep
    # the agents queued up as much as possible (working capacity is 2, so there will be a long queue).
    # Meanwhile, the agents will randomly change priority while in queue.

    random.seed(42)

    num_agents = 100

    # Add a bunch of cargo at each of the 3 airports (should be at least one per airplane)
    cargo_counter = itertools.count()
    cargo_info = []
    for _ in range(num_agents):
        cargo_info.append(CargoInfo(id=next(cargo_counter), source_airport_id=1, end_airport_id=2))
        cargo_info.append(CargoInfo(id=next(cargo_counter), source_airport_id=2, end_airport_id=3))
        cargo_info.append(CargoInfo(id=next(cargo_counter), source_airport_id=3, end_airport_id=1))
    env = generate_environment(num_of_airports=3,
                               num_of_agents=num_agents,
                               processing_time=5,
                               working_capacity=2,
                               malfunction_generator=NoEventIntervalGen(),
                               cargo_generator=HardcodedCargoGenerator(cargo_info),
                               max_cycles=300)
    obs = env.reset(seed=897)
    state = obs["a_0"]["globalstate"]

    # For each airport randomly assign a cargo at that airport to each airplane
    cargo_assignment = {}
    for airport in [1, 2, 3]:
        cargo_at_airport = [c for c in state["active_cargo"] if c.location == airport]
        agents_at_airport = [a for a, o in obs.items() if o["current_airport"] == airport]
        random.shuffle(cargo_at_airport)
        random.shuffle(agents_at_airport)
        for a, c in zip(agents_at_airport, cargo_at_airport):
            cargo_assignment[a] = c

    _done = False
    while not _done:
        actions = {}
        for a, o in obs.items():
            c = cargo_assignment[a]

            if oh.needs_orders(o):
                # If the airplane is idle, assign a new action to go back into processing (loading/unloading its cargo as appropriate)
                actions[a] = {"priority": None,
                              "cargo_to_load": [c.id] if c.id not in o["cargo_onboard"] else [],
                              "cargo_to_unload": [c.id] if c.id in o["cargo_onboard"] else [],
                              "destination": NOAIRPORT_ID,
                              }
            elif o["state"] == PlaneState.WAITING:
                # If the agent is in queue, there's a 10% chance that it will randomly change to one of three priorities
                if random.choices([True, False], weights=[0.1, 0.9]):
                    actions[a] = o["next_action"]
                    actions[a]["priority"] = random.choice([1, 2, 3])
            else:
                actions[a] = None

        obs, rewards, dones, _ = env.step(actions)
        _done = all(dones.values())

def agent_actions(env, obs):
    done = False
    cargo = 0
    prio = 1

    while not done:
        available_routes = obs['a_0']['available_routes']
        route = np.random.choice(available_routes)
        actions = {a: None for a in env.agents}

        for key in actions.keys():
            if obs['a_0']['cargo_at_current_airport']:
                actions[key] = {'priority': prio, 'cargo_to_load': [cargo], 'cargo_to_unload': set(),
                                'destination': route}
                cargo += 1
                if prio <= 3:
                    prio += 1
            elif obs['a_0']['current_airport'] == 1:
                actions[key] = {'priority': None, 'cargo_to_load': set(), 'cargo_to_unload': [cargo],
                                'destination': route}
            else:
                actions[key] = {'priority': None, 'cargo_to_load': set(), 'cargo_to_unload': set(),
                                'destination': 1}

        # print(obs)
        obs, rewards, dones, _ = env.step(actions)
        done = all(dones.values())
