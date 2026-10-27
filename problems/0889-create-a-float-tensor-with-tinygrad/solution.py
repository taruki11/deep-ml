from tinygrad import Tensor, dtypes

def to_float_tensor(values):
    """
    Converts a Python list (or list of lists) of numbers into a float32 tinygrad Tensor.
    
    Args:
        values: list or list of lists containing numbers (int/float)
        
    Returns:
        Tensor: tinygrad Tensor with dtype=dtypes.float32
    """
    return Tensor(values, dtype=dtypes.float32)