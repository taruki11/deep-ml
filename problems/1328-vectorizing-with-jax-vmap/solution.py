import jax
import jax.numpy as jnp

def dot(a, b):
    """Dot product of two 1-D vectors (given)."""
    return jnp.dot(a, b)

import jax

def batched_dot(A, B):
    return jax.vmap(dot)(A, B)