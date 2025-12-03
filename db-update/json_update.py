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

# Import TextGrad predict function
sys.path.append('prompt-optimizer')
try:
    from textgrad_optimizer import predict as textgrad_predict
    TEXTGRAD_AVAILABLE = True
except ImportError:
    TEXTGRAD_AVAILABLE = False
    print("[json_update] WARNING: Could not import textgrad_optimizer. Falling back to direct DPO.")

# Maximum number of steps the episode will run
max_cycles = 5000

# Configure OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Default model tag for DPO
DEFAULT_MODEL_TAG = "clarifier with solutions"


def load_dpo_model(model_id:str) -> str:
    """
    Load the fine-tuned DPO model name from saved metadata.
    
    Note: This function is now primarily used as a fallback when TextGrad
    is not available. The preferred path is through textgrad_predict().
    """
    meta_path = "prompt-optimizer/model_meta.json"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get(model_id).get("fine_tuned_model")
    except Exception:
        return None


def optimize_prompt_with_textgrad(prompt: str) -> str:
    """
    Optimize a prompt using TextGrad (runs a single epoch of training).
    
    This function calls textgrad_predict() which:
    1. Uses the input prompt as starting point
    2. Runs ONE epoch of TextGrad training
    3. Returns the optimized prompt
    
    Args:
        prompt: The user's raw prompt to optimize/clarify
    
    Returns:
        Optimized/clarified prompt string
    
    Fallback Behavior:
        If TextGrad is not available, falls back to direct DPO model call.
    """
    if TEXTGRAD_AVAILABLE:
        try:
            # Use TextGrad predict - runs single epoch training
            optimized = textgrad_predict(prompt)
            return optimized
        except Exception as e:
            print(f"[json_update] TextGrad prediction failed: {e}")
            # Fall through to DPO fallback
    
    # Fallback to direct DPO call
    dpo_model = load_dpo_model(DEFAULT_MODEL_TAG)
    if dpo_model:
        return clarify_with_dpo(prompt, dpo_model)
    
    # No optimization available
    return prompt


def clarify_with_dpo(prompt: str, dpo_model: str) -> str:
    """
    Use the fine-tuned DPO model directly to clarify prompts.
    
    Note: This function is now a fallback. The preferred path is through
    optimize_prompt_with_textgrad() which uses pre-trained TextGrad weights.
    
    The DPO model has been trained on preferred/non-preferred prompt pairs
    to understand what makes a good prompt clarification.
    """
    system_prompt = (
        "You are a database command interpreter. Transform ambiguous user requests "
        "into precise, actionable instructions. Make it clear whether the user wants to "
        "extract information from the database OR modify the database. Include specific "
        "field names (route_available, cost, time) and conditions. Do not make SQL, just create natural language instructions."
    )
    try:
        response = client.chat.completions.create(
            model=dpo_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DPO clarification failed: {e}")
        return prompt  # fallback to original


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

def handle_prompt(prompt: str, data_path: str, confidence_threshold: float = 0.5) -> str:
    """
    Handle a user prompt based on classification with TextGrad optimization.
    
    Classification Labels:
        A -> Extract information from database (uses GPT with JSON context)
        B -> Modify the database (triggers JSON edit pipeline)
        C -> Uncertain (requires clarification)
    
    TextGrad Integration:
    ---------------------
    When the prompt is uncertain (label C) or has low confidence, we use
    TextGrad to optimize the prompt (runs a single epoch of training).
    
    Args:
        prompt: User's natural language prompt
        data_path: Path to the JSON database file
        confidence_threshold: Minimum confidence to accept classification (default: 0.5)
    
    Returns:
        Response string based on classification result
    """
    # First classify the raw prompt
    label, confidence = classify_prompt(prompt)
    print(f"Initial classification: {label} (confidence: {confidence:.2f})")

    # If uncertain (C) OR if confidence is low for A/B, optimize with TextGrad
    if label == "C" or confidence < confidence_threshold:
        if TEXTGRAD_AVAILABLE:
            # Use TextGrad predict (runs single epoch training)
            if label == "C":
                print(f"[TextGrad] Prompt unclear (label C). Running optimization...")
            else:
                print(f"[TextGrad] Low confidence ({confidence:.2f} < {confidence_threshold}). Running optimization...")
            
            optimized_prompt = optimize_prompt_with_textgrad(prompt)
            print(f"[TextGrad] Optimized prompt: {optimized_prompt}")
        else:
            # Fallback to direct DPO model call
            dpo_model = load_dpo_model(model_id=DEFAULT_MODEL_TAG)
            if dpo_model:
                if label == "C":
                    print(f"[DPO Fallback] Prompt unclear. Using DPO model ({dpo_model}) to clarify...")
                else:
                    print(f"[DPO Fallback] Low confidence. Using DPO model ({dpo_model}) to optimize...")
                
                optimized_prompt = clarify_with_dpo(prompt, dpo_model)
                print(f"[DPO Fallback] Optimized prompt: {optimized_prompt}")
            else:
                print("[Warning] No TextGrad or DPO model available.")
                if label == "C":
                    return "I'm not sure what you want, please clarify."
                print("Proceeding with original prompt despite low confidence.")
                optimized_prompt = prompt
        
        # Re-classify the optimized prompt
        label, confidence = classify_prompt(optimized_prompt)
        print(f"Re-classified as: {label} (confidence: {confidence:.2f})")
        
        # Use the optimized prompt for subsequent processing
        prompt = optimized_prompt

    if label == "A":
        # Load JSON database and answer the query
        with open(data_path, "r") as f:
            database = json.load(f)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use only the provided database."},
                {"role": "user", "content": f"Database:\n{json.dumps(database, indent=2)}\n\nQuestion: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()
    elif label == "B":
        # Modify DB → triggers the JSON edit pipeline
        return "Database Updated."
    else:
        # Still uncertain after optimization (edge case)
        return "I'm not sure what you want, please clarify."

def openai_json_edit(in_path, out_path, user_prompt, model="gpt-5-mini", 
                     confidence_threshold=0.5):
    """
    End-to-end pipeline for JSON editing (load -> LLM call -> write) with OpenAI.
    
    This function integrates TextGrad prompt optimization:
    1. Classify the user prompt (A: extract, B: modify, C: uncertain)
    2. If uncertain or low confidence, optimize prompt via TextGrad (single epoch)
    3. Execute the appropriate action based on classification
    
    Args:
        in_path (str): Path to the input JSON file.
        out_path (str): Path to the output JSON file.
        user_prompt (str): Natural-language instruction describing the edit.
        model (str): OpenAI model for JSON editing (default: gpt-5-mini)
        confidence_threshold (float): Minimum confidence to accept classification.
    
    Returns:
        Response string from handle_prompt
    """
    response = handle_prompt(user_prompt, in_path, 
                            confidence_threshold=confidence_threshold)
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

    user_prompt = "Analyze routes from airport 3 are consistent and cross check inconsistent prices and cargoes."

    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--Test', '-T', type=int, default=0, help='input json Test to load from')
    parser.add_argument('--Level', '-L', type=int, default=0, help='input json Level to load from')
    parser.add_argument('--model', type=str, default="gpt-5-mini", help='OpenAI model to use for JSON editing')
    parser.add_argument('--instruction', '-i', type=str, default=user_prompt, help='natural language instruction to edit the input json')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='minimum confidence threshold for classification (default: 0.5)')
    args = parser.parse_args()

    in_path = f"./database/example.json"
    out_path = f"./database/updated_database.json"
    in_pkl = f"./database/example.pkl"
    solution_path = "./solution/updated_solution.json"
   
    openai_json_edit(in_path, out_path, args.instruction, args.model, 
                     confidence_threshold=args.confidence_threshold)
    write_solution(in_pkl)
    