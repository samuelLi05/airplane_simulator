# jsonl_task_loader.py

import json
import random
from typing import List, Tuple, Optional, Iterator

def load_jsonl(path: str) -> List[dict]:
    """
    Load a .jsonl file into a list of dicts.
    Each line should be a JSON object.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def prepare_pairs(
    data: List[dict],
    prompt_key: str = "prompt",
    solution_key: str = "solution",
) -> List[Tuple[str, str]]:
    """
    Convert loaded JSON list into (input_text, target_text) pairs.
    """
    pairs = []
    for obj in data:
        p = obj.get(prompt_key)
        s = obj.get(solution_key)
        if p is None or s is None:
            # skip invalid entries (or raise)
            continue
        pairs.append((p, s))
    return pairs

def split_data(
    pairs: List[Tuple[str, str]],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split (train, val, test) by fractions.
    Ensures they sum to <= 1.
    Returns triples of lists.
    """
    if seed is not None:
        random.seed(seed)
    if shuffle:
        random.shuffle(pairs)
    n = len(pairs)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    # slices
    val = pairs[:n_val]
    test = pairs[n_val : n_val + n_test]
    train = pairs[n_val + n_test :]
    return train, val, test

def batch_iter(
    pairs: List[Tuple[str, str]],
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator[List[Tuple[str, str]]]:
    """
    Yield batches of (input, target) pairs.
    """
    if seed is not None:
        random.seed(seed)
    if shuffle:
        random.shuffle(pairs)
    for i in range(0, len(pairs), batch_size):
        yield pairs[i : i + batch_size]

# Example usage:

if __name__ == "__main__":
    # Example: load, split, iterate
    all_data = load_jsonl("./prompt-optimizer/samples.jsonl")
    pairs = prepare_pairs(all_data, prompt_key="prompt", solution_key="solution")
    train, val, test = split_data(pairs, val_fraction=0.1, test_fraction=0.1, seed=42)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    for batch in batch_iter(train, batch_size=8, shuffle=True, seed=42):
        # each batch is a list of (input_text, target_text) tuples
        # feed this into your TextGrad loop
        print(batch)
        break
