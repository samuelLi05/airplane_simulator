# Environment
from airlift.envs.airlift_env import AirliftEnv

import json
import os
import networkx as nx

"""
Utility functions to encode and decode Airlift env global state objects from json and pkl formats
"""

def load_pickle_into_json(filepath:str, json_file_path:str, save_json = False):
    # recreate example.json file 
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found at: {filepath}")
    env = AirliftEnv.load(filename=filepath)
    # get agent name and get the global state
    obs = env.reset()
    state = obs[env.agents[0]]["globalstate"]

    # Manually construct the condensed dictionary
    condensed_state = {}

    # Route Map and Airports
    # We'll just take the first plane type's graph for simplicity, as per the manual generator
    plane_type_id = list(state['route_map'].keys())[0]
    graph = state['route_map'][plane_type_id]
    
    # Create a jsonpickle-like structure for the graph
    routemap_json = {
        "json://0": {
            "py/object": "networkx.classes.digraph.DiGraph",
            "_node": {f"json://{n}": data for n, data in graph.nodes(data=True)},
            "_adj": {f"json://{u}": {f"json://{v}": data for v, data in adj.items()} for u, adj in graph.adjacency()}
        }
    }
    condensed_state['route_map'] = routemap_json

    # Plane Types
    condensed_state['plane_types'] = state['plane_types']

    # Agents
    condensed_state['agents'] = state['agents']

    # Active Cargo
    #condensed_state['active_cargo'] = state['active_cargo']
    active_cargo_list = []
    for cargo_array in state['active_cargo']:
        cargo_dict = {
            "py/object": "airlift.envs.airlift_env.CargoObservation",
            "id": int(cargo_array[0]),
            "location": int(cargo_array[1]),
            "destination": int(cargo_array[2]),
            "weight": int(cargo_array[3]),
            "earliest_pickup_time": int(cargo_array[4]),
            "is_available": bool(cargo_array[5]),
            "soft_deadline": int(cargo_array[6]),
            "hard_deadline": int(cargo_array[7])
        }
        active_cargo_list.append(cargo_dict)
    condensed_state['active_cargo'] = active_cargo_list

    # Encode here with editable data format
    with open(json_file_path, 'w') as f:
        json.dump(condensed_state, f, indent=4)
    print(f"Created {json_file_path}")

def serialize_to_json(env:AirliftEnv, json_file_path:str):
    # This function is not used by the main script, but we can leave it as is
    import jsonpickle
    json_env = jsonpickle.encode(env, indent=4, keys=True)
    with open(json_file_path, 'w') as f:
        f.write(json_env)
    print(f"Created {json_file_path}")


def load_env_from_json(filepath:str):
    # This function should now use the ManualWorldGenerator
    from airlift.envs.generators.manual_generator import ManualWorldGenerator
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    
    #TODO
    return 
    # env = AirliftEnv(world_generator=ManualWorldGenerator(filepath))
    # print (f"Successfully loaded environment from {filepath}")
    # return env

if __name__ == "__main__":

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_envs_dir = os.path.join(parent_dir, "test-environments")

    for subfolder in os.listdir(test_envs_dir):
        subfolder_path = os.path.join(test_envs_dir, subfolder)

        if not os.path.isdir(subfolder_path) or "_json" in subfolder:
            continue  # skip files and json dirs, only process pickle dirs

        # Step 3: Create a parallel JSON directory (e.g., Test0_json)
        json_subfolder = os.path.join(test_envs_dir, f"{subfolder}_json")
        os.makedirs(json_subfolder, exist_ok=True)

        # Step 4: Convert all .pkl files inside subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".pkl"):
                pickle_path = os.path.join(subfolder_path, filename)
                json_filename = filename.replace(".pkl", ".json")
                json_path = os.path.join(json_subfolder, json_filename)

                try:
                    load_pickle_into_json(pickle_path, json_path)
                except Exception as e:
                    print(f"Failed to convert {pickle_path}: {e}")
                else:
                    print(f"Converted {pickle_path} to {json_path}")
