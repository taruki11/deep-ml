import numpy as np
def attention_dropout(attn_weights, values, dropout_rate: float, mask) -> list[list[float]]:
    attn_weights = np.asarray(attn_weights, dtype=float)
    values = np.asarray(values, dtype=float)
    
    # 1. 如果 dropout_rate 为 0，不进行任何丢弃与缩放
    if dropout_rate == 0:
        context = attn_weights @ values
    else:
        mask = np.asarray(mask, dtype=float)
        scale = 1.0 / (1.0 - dropout_rate)
        
        dropped_attn = attn_weights * mask * scale
        
        context = dropped_attn @ values
        
    return context.tolist()