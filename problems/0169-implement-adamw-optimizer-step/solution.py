import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    w=np.asarray(w,dtype=float)
    g=np.asarray(g,dtype=float)
    m=np.asarray(m,dtype=float)
    v=np.asarray(v,dtype=float)
    m_t = beta1 * m + (1.0 - beta1) * g
    v_t = beta2 * v + (1.0 - beta2) * (g ** 2)
    m_hat = m_t / (1.0 - beta1 ** t)
    v_hat = v_t / (1.0 - beta2 ** t)
    w_decayed = w - lr * weight_decay * w
    w_new = w_decayed - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return w_new, m_t, v_t