from openai import OpenAI
from dotenv import load_dotenv
import argparse
import prompts
import json
import sys
import os
import math

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


def classify_prompt(prompt: str) -> tuple[str, float]:
    """
    Classify a prompt as 'A', 'B', or 'C' using GPT-5-nano log probabilities.

    Returns:
        (label, confidence): 
            label = 'A' | 'B' | 'C'
            confidence = probability of the chosen label
    """
    response = client.completions.create(
        model="gpt-4o-mini",
        prompt=(
            "Classify the following user prompt into one of these options:\n"
            "A. Extract info from a database\n"
            "B. Modify the database\n"
            "C. Uncertain\n\n"
            f"Prompt: {prompt}\n\nPlease answer with a signle, capital letter:"
        ),
        max_tokens=1,
        logprobs=3,     # request logprobs for top tokens
        temperature=0   # deterministic output
    )

    choice = response.choices[0]
    
    logprobs = choice.logprobs.top_logprobs[0]

    prob_a = math.exp(logprobs.get(" A", float("-inf")))
    prob_b = math.exp(logprobs.get(" B", float("-inf")))
    prob_c = math.exp(logprobs.get(" C", float("-inf")))

    total = prob_a + prob_b + prob_c
    if total == 0:
        return "C", 0.33  # fallback

    probs = {
        "A": prob_a / total,
        "B": prob_b / total,
        "C": prob_c / total,
    }

    label = max(probs, key=probs.get)
    confidence = probs[label]

    return label, confidence

def handle_prompt(prompt: str, data_path: str) -> str:
    """
    Handle a user prompt based on classification:
    A -> Use GPT-5-nano with a JSON database to answer
    B -> Return empty string
    C -> Return clarification request
    """
    # First classify
    label, _ = classify_prompt(prompt)

    if label == "A":
        # Load JSON database
        with open(data_path, "r") as f:
            database = json.load(f)

        # Ask GPT-5-nano to answer using the database
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use only the provided database."},
                {"role": "user", "content": f"Database:\n{json.dumps(database, indent=2)}\n\nQuestion: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()
    elif label == "B":
        # Modify DB → not handled by GPT, so return nothing
        return "Database Updated."
    else:
        # Clarification
        return "I'm not sure what you want, please clarify."

def openai_json_edit(in_path, out_path, user_prompt, model="gpt-5-mini"):
    """
    End-to-end pipeline for JSON editing (load -> LLM call -> write) with OpenAI.

    params
    in_path (str): Path to the input JSON file.
    out_path (str): Path to the output JSON file.
    user_prompt (str): Natural-language instruction describing the edit.

    returns a response
    """
    response = handle_prompt(user_prompt, in_path)
    if response == "Database Updated.":
        original_json = load_json_from_file(in_path)
        request = openai_request(original_json, user_prompt, model)
        updated_json = json.loads(request.output[1].content[0].text)
        write_json_to_file(updated_json, out_path)
    return response

def write_solution(in_pkl):
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

if __name__ == '__main__':
    """
    Updated JSON will be saved to ./database/updated_database.json
    """
    user_prompt = "The route between airport 1 and airport 12 is unavailable."

    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--Test', '-T', type=int, default=0, help='input json Test to load from')
    parser.add_argument('--Level', '-L', type=int, default=0, help='input json Level to load from')
    parser.add_argument('--model', type=str, default="gpt-5-mini", help='OpenAI model to use')
    parser.add_argument('--instruction', '-i', type=str, default=user_prompt, help='natural language instruction to edit the input json')
    args = parser.parse_args()

    in_path = f"./database/example.json"
    out_path = f"./database/updated_database.json"
    in_pkl = f"./database/example.pkl"
    solution_path = "./solution/updated_solution.json"
   
    openai_json_edit(in_path, out_path, args.instruction, args.model)
    write_solution(in_pkl)
    