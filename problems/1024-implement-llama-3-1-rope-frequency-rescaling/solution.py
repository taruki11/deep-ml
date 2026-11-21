import math
import numpy as np


def llama3_rope_rescale(
    inv_freq,
    original_context_length,
    low_freq_factor,
    high_freq_factor,
    scaling_factor,
):
  """Apply Llama 3.1 style piecewise rescaling to a set of RoPE inverse frequencies.

  Args:
      inv_freq: list or array of base inverse frequencies
      original_context_length: int, the original context length used during
        pretraining
      low_freq_factor: float, threshold factor defining the low-frequency cutoff
      high_freq_factor: float, threshold factor defining the high-frequency
        cutoff
      scaling_factor: float, the factor by which low-frequency components are
        divided

  Returns:
      list of floats: adjusted inverse frequencies, each rounded to 6 decimals
  """
  low_freq_wavelen = original_context_length / low_freq_factor
  high_freq_wavelen = original_context_length / high_freq_factor

  new_freqs = []
  for freq in inv_freq:
    freq_val = float(freq)
    wavelen = 2.0 * math.pi / freq_val

    if wavelen < high_freq_wavelen:
      # 高频区：旋转充分，保持原频率不变
      new_freq = freq_val
    elif wavelen > low_freq_wavelen:
      # 低频区：波长过长，按 scaling_factor 等比压缩频率
      new_freq = freq_val / scaling_factor
    else:
      # 中频区：按预训练期内的旋转圈数进行线性平滑过渡
      smooth = (original_context_length / wavelen - low_freq_factor) / (
          high_freq_factor - low_freq_factor
      )
      new_freq = (1.0 - smooth) * (
          freq_val / scaling_factor
      ) + smooth * freq_val

    new_freqs.append(round(new_freq, 6))

  return new_freqs
