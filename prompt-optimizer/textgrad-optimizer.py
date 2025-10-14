import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
load_dotenv()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(int(y), requires_grad=False, role_description="correct answer for the query")
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

if __name__=="__main__":
    set_seed(12)
    llm_api_eval = tg.get_engine(engine_name="experimental:gpt-5-mini")
    llm_api_test = tg.get_engine(engine_name="experimental:gpt-5-mini")
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load the data and the evaluation function
    train_set, val_set, test_set, eval_fn = load_task("GSM8K_DSPy", evaluation_api=llm_api_eval)
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
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
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

