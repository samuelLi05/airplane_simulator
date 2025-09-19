# Environment
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs import PlaneType
from airlift.envs.generators.map_generators import PlainMapGenerator
from airlift.envs.generators.manual_generator import ManualWorldGenerator

# Generators
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.cargo_generators import StaticCargoGenerator

# Dynamic events
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator

# Starter kit solution
from solution.mysolution import MySolution

# Helper methods
from airlift.solutions import doepisode
from eval_solution import write_results

# Maximum number of steps the episode will run
max_cycles = 5000

# Use a plain map (this is faster to generate and captures essential elements of the scenario)
map_generator=PlainMapGenerator()

"""
Create an AirliftEnv using all the generators. There exist multiple generators for each aspect. For example instead of using the
DynamicCargoGenerator we can also use the StaticCargoGenerator.
"""



## Load from Pickle file
env = AirliftEnv.load("./test-environments/Test_1/Level_1.pkl")
obs1 = env.reset()

"""
Run a single episode utilizing the solution and manually injected JSON edits to the environment. 
"""
env_info, metrics, time_taken, total_solution_time, step_metrics = \
  doepisode(env,
            solution=MySolution(),
            render=True,
            render_sleep_time=0, # Set this to 0.1 to slow down the simulation
            env_seed=100,
            solution_seed=200,
            capture_metrics=True, 
            early_exit=False,
            inject_path= "./test-environments/Test_1_json/example.json", # edited json file we inject
            json_file_path="./test-environments/solution_example.json") # Where solution is stored

print("Missed Deliveries: {}".format(metrics.missed_deliveries))
print("Lateness:          {}".format(metrics.total_lateness))
print("Total flight cost: {}".format(metrics.total_cost))
print("Score:             {}".format(metrics.score))

write_results(env_info, step_metrics)

