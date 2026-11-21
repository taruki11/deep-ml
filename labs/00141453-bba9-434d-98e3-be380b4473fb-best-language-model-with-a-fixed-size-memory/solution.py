# --- cell 1 ---
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLM(nn.Module):
  """基于两层 GRU 的字符级循环神经网络语言模型

  总状态维度 S = num_layers * hidden_dim = 2 * 768 = 1536 float32 (6144 字节，远低于 128
  KiB 上限)
  """

  def __init__(
      self, vocab_size=96, embed_dim=128, hidden_dim=768, num_layers=2
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.gru = nn.GRU(
        embed_dim, hidden_dim, num_layers=num_layers, batch_first=True
    )
    self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x, h=None):
    # x: (batch_size, seq_len)
    # h: (num_layers, batch_size, hidden_dim)
    emb = self.embed(x)
    out, h_next = self.gru(emb, h)
    logits = self.fc(out)
    return logits, h_next


def train(text: str, alphabet: str, budget_seconds: float):
  """在线实时训练函数，严格根据 budget_seconds 时间预算自适应退出"""
  start_time = time.time()
  torch.manual_seed(42)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # 1. 建立字符词表映射
  char_to_idx = {c: i for i, c in enumerate(alphabet)}
  vocab_size = len(alphabet)
  fallback_idx = char_to_idx.get('?', 0)

  # 2. 文本向量化 (一次性常驻 GPU)
  indices = [char_to_idx.get(c, fallback_idx) for c in text]
  data = torch.tensor(indices, dtype=torch.long, device=device)
  n_tokens = len(data)

  # 3. 初始化模型与优化器
  embed_dim = 128
  hidden_dim = 768
  num_layers = 2

  model = CharLM(
      vocab_size=vocab_size,
      embed_dim=embed_dim,
      hidden_dim=hidden_dim,
      num_layers=num_layers,
  ).to(device)
  model.train()

  lr_max = 2e-3
  lr_min = 1e-4
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=lr_max, weight_decay=1e-2
  )

  # 4. 安全时间管理 (留出 8% 或至少 1.5 秒裕量避免评测超时判定 99.0)
  safety_margin = max(1.5, budget_seconds * 0.08)
  end_time = start_time + budget_seconds - safety_margin
  total_train_time = max(0.1, end_time - start_time)

  # 5. 根据语料规模适配 batch 与窗口大小
  if n_tokens < 5000:
    seq_len = min(64, max(8, n_tokens // 4))
    batch_size = 4
  else:
    seq_len = 384
    batch_size = 64

  offsets = torch.arange(seq_len + 1, device=device)

  # 6. 时间驱动的主训练循环
  while time.time() < end_time:
    # 余弦学习率衰减
    elapsed = time.time() - start_time
    progress = min(1.0, max(0.0, elapsed / total_train_time))
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
      g['lr'] = lr

    # 高效随机切片取样
    if n_tokens > seq_len + 1:
      starts = torch.randint(
          0, n_tokens - seq_len - 1, (batch_size,), device=device
      )
      idx = starts.unsqueeze(1) + offsets
      batch = data[idx]
      x = batch[:, :-1]
      y = batch[:, 1:]
    else:
      x = data[:seq_len].unsqueeze(0)
      y = data[1 : seq_len + 1].unsqueeze(0)

    optimizer.zero_grad()
    logits, _ = model(x, None)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

  # 切换为评估模式，确保输出完全确定
  model.eval()
  return model


def init_state(model, batch_size: int):
  """为 batch_size 篇新文档初始化隐状态

  返回严格的 2 维张量 (batch_size, S)
  """
  device = next(model.parameters()).device
  s_dim = model.num_layers * model.hidden_dim
  return torch.zeros((batch_size, s_dim), dtype=torch.float32, device=device)


def step(model, state, prev):
  """单字符推理步进：

  prev[b] 为前一字符索引 (起始时为 -1).
  返回:
      logprobs: (batch_size, 96) 严格归一化的对数概率
      new_state: (batch_size, S) 保持文档独立的定长隐状态
  """
  device = next(model.parameters()).device
  batch_size = len(prev)

  # 1. 确保 state 为对应设备上的 PyTorch Tensor
  if not isinstance(state, torch.Tensor):
    state = torch.from_numpy(state).to(device=device, dtype=torch.float32)
  elif state.device != device:
    state = state.to(device)

  prev_t = torch.tensor(prev, dtype=torch.long, device=device)

  # 2. 识别并重置新文档行 (prev == -1)
  is_start = prev_t == -1
  if is_start.any():
    state = state.clone()
    state[is_start] = 0.0  # 起始文档清空历史记忆
    prev_t = prev_t.clamp(min=0)  # 起始输入映射为换行符 '\n' (索引 0)

  # 3. 2D 状态展平转为 GRU 所需布局 (num_layers, batch_size, hidden_dim)
  h = (
      state.view(batch_size, model.num_layers, model.hidden_dim)
      .permute(1, 0, 2)
      .contiguous()
  )

  # 4. 单步前向计算
  x = prev_t.unsqueeze(1)  # (batch_size, 1)
  with torch.no_grad():
    emb = model.embed(x)
    out, h_next = model.gru(emb, h)
    logits = model.fc(out.squeeze(1))  # (batch_size, vocab_size)
    logprobs = F.log_softmax(logits, dim=-1)

  # 5. 将新状态展平回 2D 布局 (batch_size, num_layers * hidden_dim)
  new_state = (
      h_next.permute(1, 0, 2)
      .reshape(batch_size, model.num_layers * model.hidden_dim)
      .contiguous()
  )

  return logprobs, new_state