import numpy as np

# # https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")

def generate_seed(np_random: np.random.Generator) -> int:
    return int(np_random.integers(2 ** 32))
