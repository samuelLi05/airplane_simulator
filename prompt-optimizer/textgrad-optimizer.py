import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm

# Workaround for Python 3.9 typing issue with aiohttp 3.13+
import sys
if sys.version_info < (3, 10):
    # Monkey patch to fix the typing issue
    import typing
    _original_remove_dups_flatten = typing._remove_dups_flatten
    def _patched_remove_dups_flatten(parameters):
        try:
            return _original_remove_dups_flatten(parameters)
        except TypeError as e:
            if "unhashable type" in str(e):
                # Fallback: convert to tuple of string representations
                return tuple(dict.fromkeys(str(p) if not isinstance(p, type) else p for p in parameters))
            raise
    typing._remove_dups_flatten = _patched_remove_dups_flatten

import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
import os
import json
load_dotenv()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

class Dataset:
    def __init__(self, seq, desc):
        self._seq = list(seq)
        self._desc = desc
    def __len__(self):
        return len(self._seq)
    def __getitem__(self, idx):
        return self._seq[idx]
    def __iter__(self):
        return iter(self._seq)
    def get_task_description(self):
        return self._desc
    
class EvalFn:
        """Simple evaluator returning a tg.Variable(1|0). Supports dict or list-style calls."""
        def _extract(self, v):
            try:
                return v.get_value()
            except Exception:
                return getattr(v, "value", v)

        def __call__(self, inputs=None):
            if isinstance(inputs, dict):
                pred = inputs.get("prediction")
                gt = inputs.get("ground_truth_answer")
            else:
                # expected [x, y, response]
                _, gt, pred = inputs
            pred_text = str(self._extract(pred)).strip()
            gt_text = str(self._extract(gt)).strip()
            match = 1 if pred_text == gt_text else 0
                # Return a Variable with a role_description (required by TextGrad Variable)
            return tg.Variable(match, requires_grad=False, role_description="evaluation result")

        def parse_output(self, var):
            try:
                return int(var.value)
            except Exception:
                try:
                    return int(var.get_value())
                except Exception:
                    return int(var)

def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    # keep label as string (samples.jsonl uses strings like 'A. Extract info...')
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)
    
def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set, max_samples: int=None):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model, max_samples=max_samples))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)

def load_dataset(data_path:str, system_prompt:str, train, val, test):
    with open(data_path, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    data = [(s.get("prompt"), s.get("solution")) for s in samples]

    # Shuffle in place
    random.shuffle(data)
    n = len(data)

    if (isinstance(train, float) or isinstance(val, float) or isinstance(test, float)) or (train + val + test) <= 1:
        tr_count = int(n * float(train))
        val_count = int(n * float(val))
    else:
        tr_count = int(train)
        val_count = int(val)
    # Ensure counts do not exceed available samples
    tr_count = max(0, min(tr_count, n))
    val_count = max(0, min(val_count, n - tr_count))

    train_data = data[:tr_count]
    val_data = data[tr_count: tr_count + val_count]
    test_data = data[tr_count + val_count:]

    train_set = Dataset(train_data, system_prompt)
    val_set = Dataset(val_data, system_prompt)
    test_set = Dataset(test_data, system_prompt)
    eval_fn = EvalFn()

    return train_set, val_set, test_set, eval_fn

if __name__=="__main__":
    set_seed(12)
    llm_api_eval = tg.get_engine(engine_name="experimental:gpt-5-mini")
    llm_api_test = tg.get_engine(engine_name="experimental:gpt-5-mini")
    tg.set_backward_engine(llm_api_eval, override=True)

    system_prompt = "Could you ensure all routes from airport 2 are operational?"

    # Load the data and the evaluation function
    train_set, val_set, test_set, eval_fn = load_dataset(data_path="prompt-optimizer/samples.jsonl", system_prompt=system_prompt, train=4, val=1, test=1)
    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()

    print (STARTING_SYSTEM_PROMPT)

    train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)


    # Testing the 0-shot performance of the evaluation engine
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                                requires_grad=True,
                                role_description="system prompt to the language model")
    model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                                requires_grad=True,
                                role_description=(
                                    "Rephrase the following system prompt so that it is immediately clear whether the user's intent is A: extract information from the database, "
                                    "or B: modify/update the database. Airports, routes, costs, and cargo are database entries. Return only the rephrased prompt text (do NOT output labels like 'A' or 'B' or any other classification token). "
                                    "The rephrased prompt should make the intent obvious from its wording."
                                ))
    model = tg.BlackboxLLM(llm_api_test, system_prompt)

    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

    results = {"test_acc": [], "prompt": [], "validation_acc": []}
    results["test_acc"].append(eval_dataset(test_set, eval_fn, model, max_samples=20))
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model, max_samples=20))
    results["prompt"].append(system_prompt.get_value())

    for epoch in range(3):
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            for (x, y) in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                response = model(x)
                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output_variable = eval_fn([x, y, response])
                losses.append(eval_output_variable)
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()

            run_validation_revert(system_prompt, results, model, eval_fn, val_set, max_samples=20)

            print("sys prompt: ", system_prompt)
            test_acc = eval_dataset(test_set, eval_fn, model, max_samples=20)
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())
            if steps == 3:
                break
    print (results)

