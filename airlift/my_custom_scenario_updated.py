import sys
import os
import time

# Add the airlift project to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'airlift')))

from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.map_generators import PerlinMapGenerator
from airlift.envs.plane_types import PlaneType
from airlift.envs.airlift_env import AirliftEnv
import gym

def create_custom_scenario():
    """
    Creates a custom scenario with a random map and more airports.
    """
    return AirliftWorldGenerator(
        plane_types=[PlaneType(id=0, model='A0', max_range=4.0, speed=0.1, max_weight=10)],
        airport_generator=RandomAirportGenerator(
            max_airports=10,
            mapgen=PerlinMapGenerator()
        ),
        route_generator=RouteByDistanceGenerator(),
        cargo_generator=StaticCargoGenerator(
            num_of_tasks=20,
            soft_deadline_multiplier=2.0,
            hard_deadline_multiplier=4.0
        ),
        airplane_generator=AirplaneGenerator(2),
        max_cycles=1000
    )

def main():
    """
    This main function shows how to use your custom scenario.
    You can move this file to the 'scenarios' folder in the 'airlift-starter-kit' directory
    and use it with the 'run_custom_scenario.py' script.
    Or, you can run this file directly to visualize the scenario.
    """
    # Create the Airlift environment with the custom scenario
    env = AirliftEnv(
        gym.make(
            "Airlift-v0",
            world_generator=create_custom_scenario(),
        )
    )

    # Reset the environment to get the initial observation
    obs = env.reset()

    # Render the environment
    print("Starting simulation... A window should open to display the simulation.")
    env.render()

    # Run the simulation for a few steps
    for _ in range(200):
        # Agents do nothing
        obs, reward, done, info = env.step({agent_id: None for agent_id in obs})
        env.render()
        if done:
            break

    print("Simulation finished. The window will close in 10 seconds.")
    time.sleep(10)
    env.close()

if __name__ == "__main__":
    main()