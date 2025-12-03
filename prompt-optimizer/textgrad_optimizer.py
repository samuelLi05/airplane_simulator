import concurrent
from dotenv import load_dotenv
from tqdm import tqdm

# Workaround for Python 3.9 typing issue with aiohttp 3.13+
import sys
if sys.version_info < (3, 10):
    import typing
    _original_remove_dups_flatten = typing._remove_dups_flatten
    def _patched_remove_dups_flatten(parameters):
        try:
            return _original_remove_dups_flatten(parameters)
        except TypeError as e:
            if "unhashable type" in str(e):
                return tuple(dict.fromkeys(str(p) if not isinstance(p, type) else p for p in parameters))
            raise
    typing._remove_dups_flatten = _patched_remove_dups_flatten

import numpy as np
import random
import os
import json
import textgrad as tg
from textgrad.tasks import load_task
from textgrad.tasks.big_bench_hard import string_based_equality_fn
from textgrad.autograd.string_based_ops import StringBasedFunction
load_dotenv()
import re
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch

DATA_PATH = os.path.join(os.path.dirname(__file__), "samples.jsonl")
MODEL_META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")
DEFAULT_MODEL_TAG = "clarifier with solutions"
BATCH_SIZE = 3
TRAIN_SAMPLES = 70
VAL_SAMPLES = 10
TEST_SAMPLES = 10


MAX_TOKENS = 150
TEMPERATURE = 0.3     

ROLE_DESCRIPTION = (
    "You are a database command interpreter. Transform ambiguous user requests "
    "into precise, actionable instructions. Make it clear whether the user wants to "
    "extract information from the database OR modify the database. Include specific "
    "field names (route_available, cost, time) and conditions. Do not include SQL, only natural language instructions."
)


def set_seed(seed=12):
    np.random.seed(seed)
    random.seed(seed)


def load_dpo_model_name(model_tag: str = DEFAULT_MODEL_TAG) -> str:
    """Load the fine-tuned DPO model name from model_meta.json."""
    if not os.path.exists(MODEL_META_PATH):
        return None
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entry = meta.get(model_tag)
        if entry:
            return entry.get("fine_tuned_model")
        return None
    except Exception:
        return None

class ConstrainedEngine:
    """
    Wrapper around a TextGrad engine that applies max_tokens and temperature
    constraints to every generation call.
    """
    def __init__(self, base_engine, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE):
        self.base_engine = base_engine
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def __call__(self, prompt, **kwargs):
        # Override with our constrained values (but allow explicit overrides)
        kwargs.setdefault('max_tokens', self.max_tokens)
        kwargs.setdefault('temperature', self.temperature)
        return self.base_engine(prompt, **kwargs)
    
    def generate(self, *args, **kwargs):
        kwargs.setdefault('max_tokens', self.max_tokens)
        kwargs.setdefault('temperature', self.temperature)
        return self.base_engine.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        # Delegate all other attributes to the base engine
        return getattr(self.base_engine, name)


def get_constrained_engine(model_name: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE):
    """
    Create a TextGrad engine with constrained max_tokens and temperature.
    """
    base_engine = tg.get_engine(engine_name=model_name)
    return ConstrainedEngine(base_engine, max_tokens=max_tokens, temperature=temperature)

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

class EmbeddingEvalScorer:
    """Embedding evaluator implemented with HuggingFace `transformers` + `torch`.

    Loads AutoTokenizer and AutoModel for the chosen model and performs mean
    pooling (attention-weighted) over last_hidden_state to produce embeddings.
    This avoids importing the `sentence-transformers` package and the extra
    dataset/trainer utilities that caused import-time failures in the env.
    """
    def __init__(self, corpus=None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None, batch_size: int = 32, device: str = None):
        self.tokenizer_cls = AutoTokenizer
        self.model_cls = AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # instantiate tokenizer/model
        self.tokenizer = self.tokenizer_cls.from_pretrained(model_name)
        self.model = self.model_cls.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._gold_embeddings = {}
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".cache", "prompt_optimizer")
        os.makedirs(self.cache_dir, exist_ok=True)

        if corpus is not None:
            self._precompute_gold_embeddings(corpus)

    def _normalize(self, v: np.ndarray):
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def _encode_batch(self, texts):
        # returns numpy array shape (len(texts), dim)
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = toks["input_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)
        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = outputs.last_hidden_state  # (bs, seq_len, dim)
            # attention-weighted mean pooling
            mask = attention_mask.unsqueeze(-1).to(self.torch.float32)
            summed = (last_hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            embeddings = summed / counts
        embeddings = embeddings.cpu().numpy()
        # normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings

    def _precompute_gold_embeddings(self, corpus):
        import hashlib
        key = hashlib.sha256("\n".join([str(x or "") for x in corpus]).encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"gold_embeds_{key}.npz")
        if os.path.exists(cache_file):
            try:
                npz = np.load(cache_file, allow_pickle=True)
                texts = npz["texts"].tolist()
                embs = npz["embs"]
                for t, e in zip(texts, embs):
                    self._gold_embeddings[t] = self._normalize(e)
                return
            except Exception:
                pass

        # encode in batches
        all_embs = []
        texts = [str(t or "") for t in corpus]
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embs = self._encode_batch(batch)
            all_embs.append(embs)
        if all_embs:
            all_embs = np.vstack(all_embs)
        else:
            all_embs = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)

        for t, e in zip(texts, all_embs):
            self._gold_embeddings[t] = self._normalize(e)

        try:
            np.savez_compressed(cache_file, texts=np.array(texts, dtype=object), embs=all_embs)
        except Exception:
            pass

    def _get_embedding(self, text: str):
        text = str(text or "")
        if text in self._gold_embeddings:
            return self._gold_embeddings[text]
        emb = self._encode_batch([text])[0]
        return self._normalize(emb)

    def _cosine_sim(self, a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _extract_text(self, v):
        try:
            return v.get_value()
        except Exception:
            return getattr(v, "value", v)

    def _scorer(self, inputs):
        pred = inputs.get("prediction")
        gt = inputs.get("ground_truth_answer")
        pred_text = str(self._extract_text(pred)).strip()
        gt_text = str(self._extract_text(gt)).strip()
        a = self._get_embedding(pred_text)
        b = self._get_embedding(gt_text)
        sim = self._cosine_sim(a, b)
        sim = max(0.0, min(1.0, sim))
        return sim

    def get_eval(self):
        parent = self

        def _loss(prediction=None, ground_truth_answer=None, **kwargs):
            if (prediction is None) and isinstance(kwargs, dict) and kwargs:
                inputs = kwargs
            else:
                inputs = {"prediction": prediction, "ground_truth_answer": ground_truth_answer}
            sim = parent._scorer(inputs)
            loss = 1.0 - sim
            return str(loss)

        return StringBasedFunction(_loss, function_purpose="embedding_semantic_similarity")

    def _extract_text(self, v):
        try:
            return v.get_value()
        except Exception:
            return getattr(v, "value", v)

    def _scorer(self, inputs):
        pred = inputs.get("prediction")
        gt = inputs.get("ground_truth_answer")
        pred_text = str(self._extract_text(pred)).strip()
        gt_text = str(self._extract_text(gt)).strip()
        a = self._get_embedding(pred_text)
        b = self._get_embedding(gt_text)
        sim = self._cosine_sim(a, b)
        sim = max(0.0, min(1.0, sim))
        return sim

    def get_eval(self):
        parent = self

        def _loss(prediction=None, ground_truth_answer=None, **kwargs):
            if (prediction is None) and isinstance(kwargs, dict) and kwargs:
                inputs = kwargs
            else:
                inputs = {"prediction": prediction, "ground_truth_answer": ground_truth_answer}
            sim = parent._scorer(inputs)
            loss = 1.0 - sim
            return str(loss)

        return StringBasedFunction(_loss, function_purpose="embedding_semantic_similarity")


def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except Exception:
        eval_output_variable = eval_fn(prediction=response, ground_truth_answer=y)

    # extract the stringified loss from the returned Variable or direct value
    try:
        raw = getattr(eval_output_variable, "value", None)
        if raw is None:
            # try other accessors
            try:
                raw = eval_output_variable.get_value()
            except Exception:
                raw = eval_output_variable
        raw_s = str(raw).strip()
        loss = float(raw_s)
    except Exception:
        # As a last resort, if parse_output exists on the eval_fn, use it
        if hasattr(eval_fn, "parse_output"):
            try:
                loss = float(eval_fn.parse_output(eval_output_variable))
            except Exception:
                loss = 1.0
        else:
            loss = 1.0

    score = 1.0 - loss
    return float(score)
    
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

def load_dataset(data_path: str, system_prompt: str):
    """Load and split the dataset using hardcoded split sizes."""
    with open(data_path, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    data = [(s.get("prompt"), s.get("gold_rephrase")) for s in samples]
    random.shuffle(data)
    
    n = len(data)
    tr_count = min(TRAIN_SAMPLES, n)
    val_count = min(VAL_SAMPLES, n - tr_count)

    train_data = data[:tr_count]
    val_data = data[tr_count:tr_count + val_count]
    test_data = data[tr_count + val_count:]

    train_set = Dataset(train_data, system_prompt)
    val_set = Dataset(val_data, system_prompt)
    test_set = Dataset(test_data, system_prompt)

    corpus = [g for (_, g) in data if g is not None]
    eval_fn = EmbeddingEvalScorer(corpus=corpus).get_eval()

    return train_set, val_set, test_set, eval_fn

def predict(prompt: str) -> str:
    """
    Optimize a prompt by running a SINGLE epoch of TextGrad training.
    
    This function:
    1. Uses the input prompt as the starting system prompt
    2. Loads training data from samples.jsonl
    3. Runs ONE epoch of TextGrad optimization
    4. Returns the optimized prompt
    
    Uses constrained engine with MAX_TOKENS and TEMPERATURE settings.
    
    Args:
        prompt: The user's raw prompt to optimize
    
    Returns:
        The optimized prompt after one epoch of training
    """
    set_seed(12)
    
    # Load DPO model or fallback to gpt-4o-mini
    dpo_model_name = load_dpo_model_name(DEFAULT_MODEL_TAG)
    if dpo_model_name is None:
        dpo_model_name = "gpt-4o-mini"
        print(f"[TextGrad] No DPO model found, using {dpo_model_name}")
    else:
        print(f"[TextGrad] Using DPO foundation: {dpo_model_name}")
    
    print(f"[TextGrad] Generation params: max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}")
    
    # Initialize TextGrad engines with constrained parameters
    llm_api = get_constrained_engine(dpo_model_name, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    tg.set_backward_engine(llm_api, override=True)
    
    # Load dataset with input prompt as starting point
    train_set, val_set, test_set, eval_fn = load_dataset(DATA_PATH, prompt)
    
    # Create trainable system prompt
    system_prompt = tg.Variable(
        prompt,
        requires_grad=True,
        role_description=ROLE_DESCRIPTION
    )
    model = tg.BlackboxLLM(llm_api, system_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm_api, parameters=[system_prompt])
    
    # Track results for validation revert
    results = {"validation_acc": [], "prompt": []}
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model, max_samples=10))
    results["prompt"].append(system_prompt.get_value())
    
    # Create data loader
    train_loader = tg.tasks.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # Run ONE epoch of training
    print(f"[TextGrad] Running single epoch optimization...")
    for steps, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        losses = []
        for x, y in zip(batch_x, batch_y):
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            try:
                eval_output = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                eval_output = eval_fn([x, y, response])
            losses.append(eval_output)
        
        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()
        
        run_validation_revert(system_prompt, results, model, eval_fn, val_set, max_samples=10)
        results["prompt"].append(system_prompt.get_value())
        
        # Stop after a few steps to keep it fast
        if steps >= 3:
            break
    
    optimized = system_prompt.get_value()
    print(f"[TextGrad] Optimized prompt: {optimized}")
    return optimized

if __name__ == "__main__":
    set_seed(12)
    
    # Hardcoded values
    EPOCHS = 3
    starting_prompt = "A blocking condition appears in the route between airport 1 and 5 that needs to be shown"
    
    # Load DPO model or fallback
    dpo_model_name = load_dpo_model_name(DEFAULT_MODEL_TAG)
    if dpo_model_name is None:
        dpo_model_name = "gpt-4o-mini"
        print(f"[TextGrad] No DPO model found, using {dpo_model_name}")
    else:
        print(f"[TextGrad] Using DPO foundation: {dpo_model_name}")
    
    print(f"[TextGrad] Generation params: max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}")
    
    # Initialize TextGrad engines with constrained parameters
    llm_api_eval = get_constrained_engine(dpo_model_name, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    llm_api_test = get_constrained_engine(dpo_model_name, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    tg.set_backward_engine(llm_api_eval, override=True)
    
    # Load dataset
    train_set, val_set, test_set, eval_fn = load_dataset(DATA_PATH, starting_prompt)
    print(f"Train/Val/Test Set Lengths: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(f"Starting system prompt: {STARTING_SYSTEM_PROMPT}")
    
    train_loader = tg.tasks.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create trainable system prompt
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description=ROLE_DESCRIPTION
    )
    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])
    
    # Track results
    results = {"test_acc": [], "prompt": [], "validation_acc": []}
    results["test_acc"].append(eval_dataset(test_set, eval_fn, model, max_samples=20))
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model, max_samples=20))
    results["prompt"].append(system_prompt.get_value())
    
    # Train for 3 epochs
    print(f"\n[TextGrad] Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            for x, y in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                response = model(x)
                try:
                    eval_output = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output = eval_fn([x, y, response])
                losses.append(eval_output)
            
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()
            
            run_validation_revert(system_prompt, results, model, eval_fn, val_set, max_samples=20)
            
            print(f"sys prompt: {system_prompt}")
            test_acc = eval_dataset(test_set, eval_fn, model, max_samples=20)
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())
            
            if steps == 3:
                break
    
    # Print final results
    final_prompt = system_prompt.get_value()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final optimized prompt: {final_prompt}")
    print(f"Final Test Accuracy: {np.mean(results['test_acc'][-1]):.4f}")
    print(f"Final Validation Accuracy: {np.mean(results['validation_acc'][-1]):.4f}")
    print("=" * 60)

