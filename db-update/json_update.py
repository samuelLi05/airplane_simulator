from openai import OpenAI
from dotenv import load_dotenv
import argparse
import prompts
import json
import sys
import os

from airlift.envs.airlift_env import AirliftEnv
# Starter kit solution
sys.path.append('airlift-starter-kit')
from solution.mysolution import MySolution

# Helper methods
from airlift.solutions import doepisode
from eval_solution import write_results

# Maximum number of steps the episode will run
max_cycles = 5000

# For quick start, see: https://colab.research.google.com/drive/1plysMDLiC4HOboFAl5ufUXehyZp4uihS?authuser=5#scrollTo=5DGkohEA92jR

# Configure OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def openai_request(original_json, user_prompt, model="gpt-5-mini") -> str:
    """
    LLM call with JSON update information to OpenAI.

    params
        original_json (dict): The original JSON data to be updated.
        user-prompt (str): Natural-language instruction describing the edit.
        model (str): The OpenAI model to use for the request.

    returns
        response (openai.Response): The response object from OpenAI
    """
    prompt = f"""
        Original JSON:
        {json.dumps(original_json, indent=2)}

        Instruction:
        {user_prompt}
    """
    print(f"\nSending request to OpenAI with model {model}...")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": prompts.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    { 
                        "type": "input_text",
                        "text": prompt,
                    },
                ]
            }
        ]
    )

    return response


def load_json_from_file(path):
    """
    Load JSON from a file.

    params
        path (str): Path to the JSON file.

    returns
        original_json (dict): The loaded JSON data.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            original_json = json.load(f)
            print(f"Loaded JSON from: {path}")
            return original_json
    except Exception as e:
        print(f"Failed to read/parse JSON from {path}: {e}", file=sys.stderr)


def write_json_to_file(json_data, path):
    """
    Write JSON to a file.

    params
        json_data (dict): The JSON data to write.
        path (str): Path to the output file.

    returns
        None
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved edited JSON to: {path}")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)


def openai_json_edit(in_path, out_path, user_prompt, model="gpt-5-mini"):
    """
    End-to-end pipeline for JSON editing (load -> LLM call -> write) with OpenAI.

    params
    in_path (str): Path to the input JSON file.
    out_path (str): Path to the output JSON file.
    user_prompt (str): Natural-language instruction describing the edit.

    returns
    None
    """
    original_json = load_json_from_file(in_path)
    request = openai_request(original_json, user_prompt, model)
    updated_json = json.loads(request.output[1].content[0].text)
    write_json_to_file(updated_json, out_path)


if __name__ == '__main__':
    """
    Updated JSON will be saved to ./test-environments/updated_database.json
    """
    user_prompt = "The route between airport 1 and airport 7 is unavailable."
    user_prompt = "Add 10 additional cargo fields mapping to the cargo structure"

    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--Test', '-T', type=int, default=0, help='input json Test to load from')
    parser.add_argument('--Level', '-L', type=int, default=0, help='input json Level to load from')
    parser.add_argument('--model', type=str, default="gpt-5-mini", help='OpenAI model to use')
    parser.add_argument('--instruction', '-i', type=str, default=user_prompt, help='natural language instruction to edit the input json')
    args = parser.parse_args()

    in_path = f"./database/example.json"
    out_path = f"./database/agent_database.json"
    in_pkl = f"./database/example.pkl"
    solution_path = "./solution/agent_solution.json"

    openai_json_edit(in_path, out_path, args.instruction, args.model)

    # Load the edited json file into Airlift environment

    env = AirliftEnv.load(in_pkl)
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
                early_exit=5,
                inject_path= out_path, # edited json file we inject
                json_file_path=solution_path) # Where solution is stored

    print("Missed Deliveries: {}".format(metrics.missed_deliveries))
    print("Lateness:          {}".format(metrics.total_lateness))
    print("Total flight cost: {}".format(metrics.total_cost))
    print("Score:             {}".format(metrics.score))

    write_results(env_info, step_metrics)

