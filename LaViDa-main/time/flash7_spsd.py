# flash7.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from SparseD_utils import create_block_mask_cached, customize_mask, create_attention_block_mask

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention, dynamic=False)

class BufferCache(dict):
    """
    用于缓存 Attention Bias 等张量的简单字典类。
    """
    pass

@dataclass
class ModelConfig:
    """
    模型配置的数据类。
    这些值是根据 LLaDA 模型的典型设置进行的模拟。
    """
    d_model: int = 4096
    n_heads: int = 32
    effective_n_kv_heads: int = 32
    rope: bool = True
    rope_theta: float = 10000.0
    rope_full_precision: bool = True
    max_sequence_length: int = 8192
    include_bias: bool = False
    init_device: str = "cuda"
    flash_attention: bool = True # 假设启用 Flash Attention
    # LayerNorm 相关配置
    layer_norm_with_affine: bool = True
    bias_for_layer_norm: bool = False
    rms_norm_eps: float = 1e-5
    # attention_layer_norm 设为 False 以匹配 q_norm/k_norm 为 None 的情况
    attention_layer_norm: bool = False

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Positional Embeddings, RoPE) 的实现。
    """
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.rope_theta = config.rope_theta
        # 预热缓存
        if config.init_device != "meta":
            self.get_rotary_embedding(config.max_sequence_length, torch.device(config.init_device))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)

# --- 全局对象和函数定义 ---

config = ModelConfig()
q_norm = None
k_norm = None
rope_cache = BufferCache()
rotary_emb = RotaryEmbedding(config, rope_cache)
attn_out = nn.Linear(
    config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
)

def _cast_attn_bias(bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
    target_dtype = input_dtype
    if bias.device.type == "cuda" and torch.is_autocast_enabled():
        target_dtype = torch.get_autocast_gpu_dtype()
    elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
        target_dtype = torch.get_autocast_cpu_dtype()
    
    if bias.dtype != target_dtype:
        bias = bias.to(target_dtype)
        bias.masked_fill_(bias == float("-inf"), torch.finfo(bias.dtype).min)
    return bias

def _scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False,
) -> torch.Tensor:
    flash_attn_func = None
    if config.flash_attention:
        try: from flash_attn import flash_attn_func
        except ImportError: pass

    if flash_attn_func is not None and attn_mask is None:
        r = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal)
        return r.transpose(1, 2)
    else:
        num_kv_heads, num_q_heads = k.size(1), q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            repeats = num_q_heads // num_kv_heads
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

# 全局状态变量，用于在 attention 函数调用之间传递状态
fine_mask = None
last = None
block_mask = None

# --- Attention 函数 ---

def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False, use_block_mask_arg: bool = None, SparseD_param: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    
    global fine_mask, last, block_mask # 声明我们将修改全局状态变量

    if SparseD_param is not None:
        now_step, whole_steps, new_generation = SparseD_param['now_step'], SparseD_param['whole_steps'], SparseD_param['new_generation']
        skip, select, block_size_param = SparseD_param['skip'], SparseD_param['select'], SparseD_param['block_size']

    B, T, C = q.size()
    dtype = k.dtype

    if q_norm is not None and k_norm is not None:
        q, k = q_norm(q).to(dtype=dtype), k_norm(k).to(dtype=dtype)

    q = q.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2)
    k = k.view(B, T, config.effective_n_kv_heads, C // config.n_heads).transpose(1, 2)
    v = v.view(B, T, config.effective_n_kv_heads, C // config.n_heads).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        k, v = torch.cat((past_key, k), dim=-2), torch.cat((past_value, v), dim=-2)

    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]

    if config.rope:
        q, k = rotary_emb(q, k)

    if attention_bias is not None:
        attention_bias = _cast_attn_bias(attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype)

    if SparseD_param is None:
        att = _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    else:
        if now_step == 0:
            fine_mask, last, block_mask = None, None, None
        
        end_time = int(whole_steps * skip) + 1
        if now_step <= end_time:
            if now_step == end_time:
                query_states, key_states = q, k
                if fine_mask is None:
                    bsz, num_heads, q_len, kv_len = query_states.size(0), query_states.size(1), query_states.size(2), key_states.size(2)
                    fine_mask = torch.zeros((bsz, num_heads, (q_len + block_size_param - 1) // block_size_param, (kv_len + block_size_param - 1) // block_size_param), dtype=torch.bool, device=query_states.device)
                    for idx in range((q_len + block_size_param - 1) // block_size_param):
                        if q_len - idx * block_size_param <= new_generation or idx == (q_len + block_size_param - 1) // block_size_param - 1:
                            if last is None: last = idx
                        query_states_reduce = query_states[:, :, idx * block_size_param:(idx + 1) * block_size_param]
                        attn_weights = torch.matmul(query_states_reduce, key_states.transpose(2, 3)) / math.sqrt(num_heads)
                        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        fine_mask_update = create_attention_block_mask(attn_weights, block_size=block_size_param, keep_ratio=select) 
                        fine_mask[:, :, idx:idx+1, :] = fine_mask_update[:, :, :, :]
                    fine_mask[:, :, :, last:] = False
                
                if block_mask is None:
                    bsz, num_heads, q_len, kv_len = query_states.size(0), query_states.size(1), query_states.size(2), key_states.size(2)
                    key_states_reduce = key_states[:, :, last * block_size_param:, :]
                    for idx in range((q_len + block_size_param - 1) // block_size_param):
                        query_states_reduce = query_states[:, :, idx * block_size_param:(idx + 1) * block_size_param]
                        attn_weights = torch.matmul(query_states_reduce, key_states_reduce.transpose(2, 3)) / math.sqrt(num_heads)
                        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        fine_mask_update = create_attention_block_mask(attn_weights, block_size=block_size_param, keep_ratio=select) 
                        fine_mask[:, :, idx:idx + 1, last:] = torch.logical_or(fine_mask[:, :, idx:idx + 1, last:], fine_mask_update[:, :, :, :])
                    
                    new_mask = customize_mask(fine_mask, block_size=block_size_param)
                    block_mask = create_block_mask_cached(new_mask, bsz, num_heads, q_len, kv_len, device=query_states.device, _compile=True)
            
            att = _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        else:
            att = flex_attention(q, k, v, block_mask=block_mask)

    att = att.transpose(1, 2).contiguous().view(B, T, C)
    return attn_out(att.float()), present


# --- 性能测试函数 ---

def test_block_sparse_attention(BS, HEAD, SEQLEN, DIM, BLOCK_SIZE, SELECT_RATIO):
    """
    分阶段测试 SparseD 的性能：
    1. 生成 block_mask + 执行标准 Attention
    2. 使用已生成的 block_mask 执行 flex_attention
    """
    print(f"\n--- 测试参数: BS={BS}, HEAD={HEAD}, SEQLEN={SEQLEN}, DIM={DIM}, BLOCK_SIZE={BLOCK_SIZE}, SELECT_RATIO={SELECT_RATIO} ---")
    dtype = torch.bfloat16
    DIM_MODEL = HEAD * DIM
    
    # 1. 准备输入数据
    q = torch.randn((BS, SEQLEN, DIM_MODEL), dtype=dtype, device="cuda")
    k = torch.randn((BS, SEQLEN, DIM_MODEL), dtype=dtype, device="cuda")
    v = torch.randn((BS, SEQLEN, DIM_MODEL), dtype=dtype, device="cuda")
    
    # 参数一：用于触发 mask 生成 (now_step = end_time)
    sparse_params_gen = {'now_step': 1, 'whole_steps': 0, 'new_generation': 64, 'skip': 0, 'select': SELECT_RATIO, 'block_size': BLOCK_SIZE}
    # 参数二：用于触发 flex_attention (now_step > end_time)
    sparse_params_flex = {'now_step': 2, 'whole_steps': 0, 'new_generation': 64, 'skip': 0, 'select': SELECT_RATIO, 'block_size': BLOCK_SIZE}
    
    print("预热 CUDA 核心并生成全局 block_mask...")
    _, _ = attention(q, k, v, SparseD_param=sparse_params_gen)
    _, _ = attention(q, k, v, SparseD_param=sparse_params_flex)
    torch.cuda.synchronize()
    print("预热完成。")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_runs = 50

    # --- 阶段一: 测试 Mask 生成 + SDPA ---
    print("\n--- 阶段一: 测试 Mask 生成 + SDPA ---")
    start_event.record()
    for _ in range(num_runs):
        global block_mask, fine_mask, last
        block_mask, fine_mask, last = None, None, None
        _, _ = attention(q, k, v, SparseD_param=sparse_params_gen)
    end_event.record()
    torch.cuda.synchronize()
    time_gen_and_sdpa = start_event.elapsed_time(end_event) / num_runs
    print(f"平均时间 (Mask 生成 + SDPA): {time_gen_and_sdpa:.4f} 毫秒")

    # 在阶段二计时前，必须先确保全局 block_mask 已被生成
    print("\n为阶段二重新生成全局 block_mask...")
    block_mask, fine_mask, last = None, None, None
    _, _ = attention(q, k, v, SparseD_param=sparse_params_gen)
    torch.cuda.synchronize()
    print("Mask 已准备就绪。")
    
    # --- 阶段二: 仅测试 Flex Attention Kernel ---
    print("\n--- 阶段二: 仅测试 Flex Attention Kernel (使用预生成 Mask) ---")
    start_event.record()
    for _ in range(num_runs):
        _, _ = attention(q, k, v, SparseD_param=sparse_params_flex)
    end_event.record()
    torch.cuda.synchronize()
    time_flex_only = start_event.elapsed_time(end_event) / num_runs
    print(f"平均时间 (仅 Flex Attention): {time_flex_only:.4f} 毫秒")
    
if __name__ == '__main__':
    # 您可以在这里修改参数进行测试
    BS, HEAD, SEQLEN, DIM = 1, 32, 8192, 128
    BLOCK_SIZE = 128
    SELECT_RATIO = 0.1 # 保留 10% 的块
    
    # 将测试参数更新到全局 config 对象
    config.n_heads = HEAD
    config.effective_n_kv_heads = HEAD
    
    test_block_sparse_attention(BS, HEAD, SEQLEN, DIM, BLOCK_SIZE, SELECT_RATIO)