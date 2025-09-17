# Environment
from airlift.envs.airlift_env import AirliftEnv

import jsonpickle
import os

def load_pickle_into_json(filepath:str, json_file_path:str, save_json = False):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found at: {filepath}")
    env = AirliftEnv.load(filename=filepath)
    json_env = jsonpickle.encode(env, indent=4, keys=True)
    # Encode here with editable data format
    with open(json_file_path, 'w') as f:
        f.write(json_env)
    print(f"Created {json_file_path}")

def serialize_to_json(env:AirliftEnv, json_file_path:str):
    # Serialize to json schema
    json_env = jsonpickle.encode(env, indent=4, keys=True)
    # Encode here with editable dat format
    with open(json_file_path, 'w') as f:
        f.write(json_env)
    print(f"Created {json_file_path}")


def load_env_from_json(filepath:str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    
    with open(filepath, 'r') as f:
        json_string = f.read()

    env = jsonpickle.decode(json_string)
    print (f"Successfully loaded environment from {filepath}")
    return env

if __name__ == "__main__":

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_envs_dir = os.path.join(parent_dir, "test-environments")

    for subfolder in os.listdir(test_envs_dir):
        subfolder_path = os.path.join(test_envs_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue  # skip files, only process dirs

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