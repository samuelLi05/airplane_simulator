import json
import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def load_samples(data_path: str) -> List[Dict]:
    """Load prompt samples from JSONL file."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def create_dpo_training_data(samples: List[Dict], system_prompt: str) -> List[Dict]:
    """Convert samples to OpenAI DPO training format.
    
    Uses the 'solution' field (A, B, or C) to determine the target classification intent:
    - A: Extract information from database (query/read)
    - B: Modify the database (update/write)
    - C: Uncertain/ambiguous (needs clarification)
    
    The gold_rephrase should clarify the prompt to match the solution's intent.
    The rejected response represents a wrong classification or unclear intent.
    """
    training_data = []
    
    for sample in samples:
        original_prompt = sample.get("prompt", "")
        gold_rephrase = sample.get("gold_rephrase", "")
        solution = sample.get("solution", "C")  # A, B, or C
        
        if not original_prompt or not gold_rephrase:
            continue
        
        # Create rejected response based on the solution classification
        if solution == "A":
            rejected_response = f"Update the database based on: {original_prompt}"
        elif solution == "B":
            rejected_response = f"Show me information about: {original_prompt.lower()}"
        else:  # solution == "C"
            rejected_response = f"The user wants to {original_prompt.lower()}"
        
        # OpenAI DPO format: input, preferred_output, non_preferred_output
        dpo_example = {
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_prompt}
                ],
            },
            "preferred_output": [
                {"role": "assistant", "content": gold_rephrase}
            ],
            "non_preferred_output": [
                {"role": "assistant", "content": rejected_response}
            ]
        }
        
        training_data.append(dpo_example)
    
    return training_data


def save_training_file(training_data: List[Dict], output_path: str):
    """Save training data to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in training_data:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(training_data)} training examples to {output_path}")


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """Upload training file to OpenAI."""
    print(f"Uploading {file_path}...")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id
    print(f"File uploaded successfully. File ID: {file_id}")
    return file_id


def save_model_metadata(model_name: str, output_dir: str, model_tag: str = "default"):
    """Save the fine-tuned model name/ID to disk under a user-specified tag."""
    path = os.path.join(output_dir, "model_meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            all_meta = json.load(f)
    else:
        all_meta = {}
    all_meta[model_tag] = {"fine_tuned_model": model_name, "saved_at": time.time()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Saved model metadata under tag '{model_tag}' to {path}")


def load_model_metadata(output_dir: str, model_tag: str = "default") -> Optional[str]:
    """Return the saved fine-tuned model name for a given tag, else None."""
    path = os.path.join(output_dir, "model_meta.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        all_meta = json.load(f)
    entry = all_meta.get(model_tag)
    if entry:
        return entry.get("fine_tuned_model")
    return None


def _load_response_cache(cache_path: str) -> Dict[str, str]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_response_cache(cache: Dict[str, str], cache_path: str):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _messages_to_key(messages: List[Dict]) -> str:
    # Stable key for a list of message dicts
    return json.dumps(messages, sort_keys=True)


def get_cached_response_or_call(client: OpenAI, model: str, messages: List[Dict], cache_path: str, **kwargs) -> str:
    """Return a cached response for messages if present, otherwise call the API and cache the output.

    This lets you avoid repeated API calls for the same prompts and is useful during
    development to avoid re-invoking the fine-tuned model for identical inputs.
    """
    cache = _load_response_cache(cache_path)
    key = _messages_to_key(messages)
    if key in cache:
        return cache[key]

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    output = response.choices[0].message.content
    cache[key] = output
    try:
        _save_response_cache(cache, cache_path)
    except Exception:
        pass
    return output


def create_dpo_finetune_job(
    client: OpenAI,
    training_file_id: str,
    model: str,
    suffix: str
) -> str:
    """Create a DPO fine-tuning job."""
    print(f"Creating DPO fine-tuning job for model: {model}")
    
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        suffix=suffix,
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": 0.1,
                    "n_epochs": 3,
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto"
                }
            }
        }
    )
    
    job_id = job.id
    print(f"Fine-tuning job created. Job ID: {job_id}")
    print(f"Status: {job.status}")
    return job_id


def monitor_finetune_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """Monitor the fine-tuning job until completion."""
    print(f"\nMonitoring job {job_id}...")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
        
        if status == "succeeded":
            print(f"\n✓ Fine-tuning completed!")
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif status in ["failed", "cancelled"]:
            print(f"\n✗ Fine-tuning {status}")
            if job.error:
                print(f"Error: {job.error}")
            return None
        
        time.sleep(poll_interval)


def test_model(client: OpenAI, model: str, test_prompts: List[str], system_prompt: str, cache_path: Optional[str] = None):
    """Test the model on sample prompts."""
    print(f"\nTesting model: {model}")
    print("=" * 80)
    
    for prompt in test_prompts:
        print(f"\nInput: {prompt}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        if cache_path:
            output = get_cached_response_or_call(
                client, model, messages, cache_path, temperature=0.3, max_tokens=150
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
            output = response.choices[0].message.content

        print(f"Output: {output}")


def main():
    # Configuration
    data_path = "prompt-optimizer/samples.jsonl"
    output_dir = "prompt-optimizer"
    model = "gpt-4.1-2025-04-14"
    suffix = "prompt-clarifier"

    model_tag = "clarifier with solutions"
    
    system_prompt = (
        "You are a database command interpreter. Transform ambiguous user requests "
        "into precise, actionable instructions. Make it clear whether the user wants to "
        "extract information from the database OR modify the database. Include specific "
        "field names (route_available, cost, time) and conditions. Do not make SQL, only natural language instructions."
    )
    
    test_prompts = [
        "Check routes from airport 7",
        "Make sure airport 3 routes are working",
        "What's the status of routes to airport 9?",
        "Fix the connection between 2 and 8",
        "Show me problematic routes"
    ]
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load and prepare training data
    print(f"Loading samples from {data_path}...")
    samples = load_samples(data_path)
    print(f"Loaded {len(samples)} samples")
    # Check if we've already saved a fine-tuned model id and reuse it
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating DPO training data...")
    training_data = create_dpo_training_data(samples, system_prompt)
    
    # Show distribution of solution types for verification
    solution_counts = {"A": 0, "B": 0, "C": 0}
    for sample in samples:
        sol = sample.get("solution", "C")
        solution_counts[sol] = solution_counts.get(sol, 0) + 1
    print(f"Training data breakdown: {solution_counts['A']} queries (A), "
            f"{solution_counts['B']} modifications (B), {solution_counts['C']} uncertain (C)")

    # Save to file
    training_file_path = os.path.join(output_dir, "dpo_training_data.jsonl")
    save_training_file(training_data, training_file_path)

    # Upload training file
    training_file_id = upload_training_file(client, training_file_path)

    # Create fine-tuning job
    job_id = create_dpo_finetune_job(client, training_file_id, model, suffix)

    # Monitor job
    finetuned_model = monitor_finetune_job(client, job_id)
    
    # Test fine-tuned model
    if finetuned_model:
        # persist metadata so future runs can skip retraining
        try:
            save_model_metadata(finetuned_model, output_dir, model_tag=model_tag)
        except Exception:
            pass

        # response cache path (optional) - caches outputs so identical prompts don't re-call the API
        cache_path = os.path.join(output_dir, "response_cache.json")
        print("\n" + "=" * 80)
        print("TESTING FINE-TUNED MODEL")
        print("=" * 80)
        test_model(client, finetuned_model, test_prompts, system_prompt, cache_path=cache_path)
        
        print(f"\n✓ Fine-tuned model ready: {finetuned_model}")


if __name__ == "__main__":
    main()
