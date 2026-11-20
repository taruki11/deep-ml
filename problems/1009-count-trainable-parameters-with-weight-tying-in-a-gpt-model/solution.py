def count_gpt_parameters(cfg: dict) -> list[int]:
  """计算 GPT 模型在无权重绑定和开启权重绑定下的可训练参数量。"""
  vocab_size = cfg["vocab_size"]
  context_length = cfg["context_length"]
  emb_dim = cfg["emb_dim"]
  n_layers = cfg["n_layers"]
  qkv_bias = cfg.get("qkv_bias", False)

  # 1. 入口词嵌入与位置嵌入
  tok_emb = vocab_size * emb_dim
  pos_emb = context_length * emb_dim

  # 2. 单个 Transformer Block 内的参数拆解
  # (a) Q, K, V 三个矩阵 (emb_dim -> emb_dim)，仅当 qkv_bias=True 时带偏置
  qkv_params = 3 * (emb_dim * emb_dim)
  if qkv_bias:
    qkv_params += 3 * emb_dim

  # (b) 注意力输出投影 (总是带 bias)
  attn_out_params = emb_dim * emb_dim + emb_dim

  # (c) 两个 LayerNorm，每个有 scale 和 shift 两个向量
  block_ln_params = 2 * (2 * emb_dim)

  # (d) FFN 两层线性层 (emb_dim -> 4*emb_dim -> emb_dim)，均带 bias
  ffn_hidden = 4 * emb_dim
  ffn_params = (emb_dim * ffn_hidden + ffn_hidden) + (
      ffn_hidden * emb_dim + emb_dim
  )

  single_block_params = (
      qkv_params + attn_out_params + block_ln_params + ffn_params
  )

  # 3. 最终 LayerNorm 与 LM Head（无 bias）
  final_ln_params = 2 * emb_dim
  lm_head_params = emb_dim * vocab_size

  # 4. 汇总结果
  total_params = (
      tok_emb
      + pos_emb
      + n_layers * single_block_params
      + final_ln_params
      + lm_head_params
  )
  # 开启 Weight Tying 时，LM Head 与 Token Embedding 共享，不再重复计算
  tied_params = total_params - lm_head_params

  return [int(total_params), int(tied_params)]