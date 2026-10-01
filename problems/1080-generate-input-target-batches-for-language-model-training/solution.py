import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    # Your code here
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, len(data) - block_size, size=batch_size)
    x = [data[i : i + block_size] for i in offsets]
    y = [data[i + 1 : i + 1 + block_size] for i in offsets]
    return np.array([x,y])