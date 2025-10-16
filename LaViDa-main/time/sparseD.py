import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum

# 设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ==============================================================================
# SECTION 1: 模型定义 (从 modeling_llada.py 提取并整合)
# 我们将所有需要的类和函数都放在这里，使其自成一体。
# ==============================================================================

# --- 配置类 ---
class StrEnum(str, Enum):
    pass

class ActivationType(StrEnum):
    swiglu = "swiglu"
    silu = "silu"

@dataclass
class ModelConfig:
    d_model: int = 4096
    n_heads: int = 32
    n_layers: int = 1 # 测试只需要一层
    mlp_ratio: int = 4
    max_sequence_length: int = 8192
    vocab_size: int = 32000
    rope: bool = True
    rope_theta: float = 10000.0
    flash_attention: bool = False
    activation_type: ActivationType = ActivationType.silu
    torch_dtype: torch.dtype = torch.bfloat16
    include_bias: bool = False
    
    # 从 LLaDABlock 提取的属性
    @property
    def mlp_hidden_size(self):
        return self.mlp_ratio * self.d_model
    
    @property
    def effective_n_kv_heads(self):
        return self.n_heads # 简化：假设没有 GQA/MQA

# --- 工具函数和模块 ---

# Mock flex_attention if not available

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

# SparseD 工具函数 (核心)
def create_attention_block_mask(scores, block_size, keep_ratio):
    batch_size, num_heads, q_len, kv_len = scores.shape
    q_blocks = (q_len + block_size - 1) // block_size
    kv_blocks = (kv_len + block_size - 1) // block_size

    scores = scores.reshape(batch_size, num_heads, q_blocks, block_size, kv_blocks, block_size)
    block_scores = scores.mean(dim=(3, 5))

    topk_k = int(keep_ratio * kv_blocks)
    _, topk_indices = torch.topk(block_scores, k=topk_k, dim=-1)
    
    mask = torch.zeros_like(block_scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_indices, value=True)
    return mask

class BufferCache(dict):
    pass

class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.normalized_shape = (config.d_model,)
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=config.torch_dtype))
    
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, weight=self.weight.float(), bias=None, eps=self.eps).to(x.dtype)

# --- 核心模型块定义 ---
class LLaDABlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.mlp_hidden_size
        
        # 简化版LayerNorm
        self.attn_norm = LayerNorm(config)
        self.ff_norm = LayerNorm(config)
        
        # 简化版线性层 (为测试创建)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (config.d_model, config.d_model, config.d_model)
        self.att_proj = nn.Linear(config.d_model, sum(self.fused_dims), bias=config.include_bias, dtype=config.torch_dtype)
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=config.include_bias, dtype=config.torch_dtype)
        
        self.ff_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, dtype=config.torch_dtype)
        self.up_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, dtype=config.torch_dtype)
        self.ff_out = nn.Linear(self.hidden_size, config.d_model, bias=config.include_bias, dtype=config.torch_dtype)
        
        self.dropout = nn.Dropout(0.0)
        
        # 用于缓存 mask 的状态
        self.block_mask = None
        self.fine_mask = None
        self.last = None

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, SparseD_param: Optional[Dict] = None):
        B, T, C = q.size()
        
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        # --- SparseD 核心逻辑 ---
        if SparseD_param is None:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            now_step = SparseD_param['now_step']
            whole_steps = SparseD_param['whole_steps']
            skip = SparseD_param['skip']
            select = SparseD_param['select']
            block_size = SparseD_param['block_size']
            
            end_time = int(whole_steps * skip) + 1

            if now_step <= end_time:
                # 阶段一：计算 Attention Score 并生成 Mask
                if now_step == end_time or self.block_mask is None:
                    print(f"   -> (Inside Attention) Step {now_step}: Generating sparse mask...")
                    bsz, num_heads, q_len, head_dim = q.shape
                    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                    
                    self.fine_mask = create_attention_block_mask(attn_weights, block_size=block_size, keep_ratio=select)
                    
                    # 简化版 mask 创建
                    self.block_mask = self.fine_mask.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
                    self.block_mask = self.block_mask[:, :, :q_len, :k.shape[2]]
                    # 转换成 flex_attention 兼容格式 (简化)
                    # 实际的 create_block_mask_cached 会更复杂, 这里仅为示意
                    self.flex_block_mask = self.fine_mask 

                att = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:
                # 阶段二：使用缓存的 Mask 进行稀疏计算
                print(f"   -> (Inside Attention) Step {now_step}: Reusing sparse mask with flex_attention...")
                if self.flex_block_mask is None:
                    raise RuntimeError("Sparse mask was not generated. Run a mask generation step first.")
                
                # B, H, T, T_kv -> B, H, Q_blks, KV_blks
                q_blks = (q.shape[2] + block_size -1) // block_size
                kv_blks = (k.shape[2] + block_size -1) // block_size
                mask_for_flex = self.flex_block_mask[:,:,:q_blks,:kv_blks]
                
                att = flex_attention(q, k, v, block_mask=mask_for_flex)

        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(att)

    def forward(self, x: torch.Tensor, SparseD_param: Optional[Dict] = None):
        og_x = x
        
        # Attention Block
        x_normed = self.attn_norm(x)
        q, k, v = x_normed, x_normed, x_normed # 简化: Q, K, V 来自同一源
        att = self.attention(q, k, v, SparseD_param=SparseD_param)
        x = og_x + self.dropout(att)
        
        # FFN Block
        og_x = x
        x_normed = self.ff_norm(x)
        gate_output = self.ff_proj(x_normed)
        value_output = self.up_proj(x_normed)
        x = F.silu(gate_output) * value_output
        x = self.ff_out(x)
        x = og_x + self.dropout(x)
        return x

# ==============================================================================
# SECTION 2: 基准测试 (Benchmark)
# ==============================================================================
def measure_time(func, num_runs=100, warm_up_runs=10):
    print(f"   -> Warming up for {warm_up_runs} iterations...")
    for _ in range(warm_up_runs):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"   -> Running benchmark for {num_runs} iterations...")
    start_event.record()
    for _ in range(num_runs):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / num_runs

def run_benchmark():
    # --- 1. 配置 ---
    BATCH_SIZE = 1
    SEQ_LENGTH = 8192
    D_MODEL = 4096
    NUM_HEADS = 32
    DTYPE = torch.bfloat16
    DEVICE = "cuda"

    print("--- SparseD Attention Standalone Benchmark ---")
    config = ModelConfig(d_model=D_MODEL, n_heads=NUM_HEADS, torch_dtype=DTYPE)
    transformer_block = LLaDABlock(layer_id=0, config=config, cache=BufferCache()).to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL, dtype=DTYPE, device=DEVICE)

    SparseD_param = {
        'skip': 0.2, 'select': 0.3, 'block_size': 128,
        'whole_steps': SEQ_LENGTH,
    }

    # --- 2. 第一次调用：测量 Mask 生成时间 ---
    print("\n--- Phase 1: Benchmarking Mask Generation ---")
    mask_gen_params = SparseD_param.copy()
    mask_gen_params['now_step'] = int(SEQ_LENGTH * SparseD_param['skip']) # 在临界点触发
    
    # 测量后，mask 会被缓存在 transformer_block 对象中
    mask_generation_time_ms = measure_time(lambda: transformer_block.forward(x, SparseD_param=mask_gen_params))

    # --- 3. 第二次调用：测量稀疏推理时间 ---
    print("\n--- Phase 2: Benchmarking Sparse Inference ---")
    inference_params = SparseD_param.copy()
    inference_params['now_step'] = int(SEQ_LENGTH * SparseD_param['skip']) + 1 # 在临界点之后
    
    sparse_inference_time_ms = measure_time(lambda: transformer_block.forward(x, SparseD_param=inference_params))
    
    # --- 4. 打印结果 ---
    print("\n\n--- Final Benchmark Results ---")
    print(f"Phase 1 (Mask Generation) Avg Time: {mask_generation_time_ms:.4f} ms")
    print(f"Phase 2 (Sparse Inference) Avg Time: {sparse_inference_time_ms:.4f} ms")
    
    if sparse_inference_time_ms > 0:
        speedup = mask_generation_time_ms / sparse_inference_time_ms
        print(f"🚀 Speedup (Phase 2 vs Phase 1): {speedup:.2f}x")
    print("---------------------------------\n")

if __name__ == '__main__':
    run_benchmark()