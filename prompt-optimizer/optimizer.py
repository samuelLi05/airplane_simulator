import textgrad as tg
import os
from jsonl_loader import load_jsonl, prepare_pairs, split_data, batch_iter
from dotenv import load_dotenv

from textgrad.loss import TextLoss
from textgrad.tasks.big_bench_hard import string_based_equality_fn
from textgrad.autograd.string_based_ops import StringBasedFunction

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def optimize_system_prompt(model_name: str = "gpt-4o-mini", starting_prompt: str = None, batch_size: int = 3, num_epochs: int = 3):
    """
    Optimize a system prompt for classifying user prompts into categories A, B, or C.
    """
    llm_api_eval = tg.get_engine(engine_name=model_name)
    llm_api_test = tg.get_engine(engine_name=model_name)
    tg.set_backward_engine(llm_api_eval, override=True)


    # Load & prepare
    all_data = load_jsonl("./prompt-optimizer/samples.jsonl")
    pairs = prepare_pairs(all_data, prompt_key="prompt", solution_key="solution")
    train_pairs, val_pairs, test_pairs = split_data(pairs, val_fraction=0.1, test_fraction=0.1, seed=123)

    # Set up system prompt, model, optimizer as before
    if starting_prompt is None:
        starting_prompt = (
                        "Classify the given user prompt into one of these options:\n"
                        "A. Extract info from a database\n"
                        "B. Modify the database\n"
                        "C. Uncertain\n\n"
                        "Please answer with a single, capital letter:"
                    )
    system_prompt = tg.Variable(starting_prompt, requires_grad=True, role_description="system prompt")
    model = tg.BlackboxLLM(llm_api_test, system_prompt=system_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])
    fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
    eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)

    # Training loop
    for epoch in range(num_epochs):
        for batch in batch_iter(train_pairs, batch_size=batch_size, shuffle=True, seed=epoch):
            optimizer.zero_grad()
            losses = []
            print(f"Epoch {epoch}, Batch {batch}:")
            for (x_text, y_text) in batch:
                x = tg.Variable(x_text, requires_grad=False, role_description="input prompt")
                y = tg.Variable(y_text, requires_grad=False, role_description="correct label")
                pred = model(x)
                loss = eval_fn(inputs={"prediction": pred, "ground_truth_answer": y})
                losses.append(loss)
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()

            print(f"System Prompt after epoch {epoch}:\n{system_prompt.get_value()}")
            print("-" * 50)

        # run validation?

    # After training, evaluate on test_pairs similarly
    return system_prompt.get_value()

system_prompt = optimize_system_prompt()


# Test the optimized system prompt
prompt = "A blocking condition appears in the route between airport 1 and 5 that needs to be shown"  # Expeted: 'B'
FULL_PROMPT = system_prompt + f"\n\nUser Prompt: [{prompt}]\n\nPlease answer with a single, capital letter."

from dotenv import load_dotenv
from openai import OpenAI
import os
import math

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def classify_prompt(prompt: str) -> tuple[str, float]:
    response = client.completions.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=1,
        logprobs=3,     # request logprobs for top tokens
        temperature=0   # deterministic output
    )

    choice = response.choices[0]
    
    logprobs = choice.logprobs.top_logprobs[0]

    prob_a = math.exp(logprobs.get(" A", float("-inf")))
    prob_b = math.exp(logprobs.get(" B", float("-inf")))
    prob_c = math.exp(logprobs.get(" C", float("-inf")))

    print(choice)
    print(f"Logprobs: A: {prob_a}, B: {prob_b}, C: {prob_c}")

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

res = classify_prompt(FULL_PROMPT)
print(res)