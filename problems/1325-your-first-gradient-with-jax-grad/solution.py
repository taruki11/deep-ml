import jax

def df_dx(x: float) -> float:
    return float(jax.grad(lambda v: v**3 + 2 * v)(float(x)))