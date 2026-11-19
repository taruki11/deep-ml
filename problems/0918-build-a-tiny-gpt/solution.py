import torch
import torch.nn as nn
class TinyGPT(nn.Module):

  def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
    super().__init__()
    # 1. 词嵌入与可学习的位置嵌入
    self.token_emb = nn.Embedding(vocab_size, d_model)
    self.pos_emb = nn.Embedding(max_seq_len, d_model)
    # 2. 堆叠 num_layers 个 Pre-LN Transformer 模块
    # 注意题目要求的参数:
    # d_model, nhead=num_heads, dim_feedforward=4*d_model,
    # activation='gelu', batch_first=True, norm_first=True, dropout=0.0
    self.blocks = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # norm_first=True 即 Pre-LN 结构
            dropout=0.0,
        )
        for _ in range(num_layers)
    ])

    # 3. 最终的 LayerNorm
    self.ln_f = nn.LayerNorm(d_model)

    # 4. 输出映射头: 将 d_model 维度映射回整个词表 (bias=False)
    self.head = nn.Linear(d_model, vocab_size, bias=False)

  def forward(self, idx):
    # idx 形状: (B, T)
    B, T = idx.shape

    # 生成位置索引: [0, 1, ..., T-1]，并保证和输入处于同一设备 (CPU/GPU)
    positions = torch.arange(0, T, device=idx.device)

    # 词向量与位置向量相加: (B, T, d_model)
    x = self.token_emb(idx) + self.pos_emb(positions)

    # 生成上三角全为 -inf 的因果掩码 (8x8)
    # 确保位置 t 只能看 0 到 t，绝不能看到 t+1 之后的未来
    mask = nn.Transformer.generate_square_subsequent_mask(
        T, device=idx.device
    )

    # 逐层穿过 Transformer Blocks
    for block in self.blocks:
      x = block(x, src_mask=mask)

    # 最终 LayerNorm
    x = self.ln_f(x)

    # 映射到词表概率分布 Logits: (B, T, vocab_size)
    logits = self.head(x)

    return logits
