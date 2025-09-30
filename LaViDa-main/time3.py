import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import math
import time
import torch.nn.functional as F
from tqdm import tqdm

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import pdb

def _eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    print_shapes: bool = True,
) -> torch.Tensor:
    
    if print_shapes:
        print("\n--- [Eager Attention] Shape Inspection ---")
        print(f"Input q.shape:          {q.shape}")
        print(f"Input k.shape:          {k.shape}")
        print(f"Input v.shape:          {v.shape}")
        print("------------------------------------------")

    head_size = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_size)
    
    if print_shapes:
        print(f"scores.shape:           {scores.shape} (q @ k.T)")

    attn_weights = F.softmax(scores, dim=-1)

    if print_shapes:
        print(f"attn_weights.shape:     {attn_weights.shape} (Softmax over scores)")

    output = torch.matmul(attn_weights, v)
    
    if print_shapes:
        print(f"output.shape:           {output.shape} (Final output tensor)")
        print("--- End of Shape Inspection ---")
    
    return output

def _request_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    decode_step: int,
    topk_update_interval: int,
    cached_topk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:

    B, nh, T_q, hs = q.shape
    T_k = k.shape[2]
    calculation_time = 0.0
    
    # We only print the shapes on the first step to avoid cluttering the output
    if decode_step == 0:
        print("\n--- [ReQuest Attention] Shape Inspection (decode_step=0) ---")
        print(f"Input q.shape:          {q.shape}")
        print(f"Input k.shape:          {k.shape}")
        print(f"Input v.shape:          {v.shape}")
        print("---------------------------------------------------------")

    effective_top_k = min(top_k, T_k)
    
    if cached_topk_indices is None or decode_step % topk_update_interval == 0:
        full_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hs)
        _, top_k_indices = torch.topk(full_scores, k=effective_top_k, dim=-1, sorted=False)
        
        if decode_step == 0:
            print(f"full_scores.shape:      {full_scores.shape} (Scores for all tokens)")
            print(f"top_k_indices.shape:    {top_k_indices.shape} (Indices of top-k keys for each query)")
            print("---------------------------------------------------------")

    else:
        top_k_indices = cached_topk_indices
    
    # Expand dimensions for the gather operation
    k_expanded = k.unsqueeze(2).expand(-1, -1, T_q, -1, -1)
    v_expanded = v.unsqueeze(2).expand(-1, -1, T_q, -1, -1)
    indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, hs)
    
    if decode_step == 0:
        print(f"k_expanded.shape:       {k_expanded.shape} (K expanded for every query pos)")
        print(f"indices_expanded.shape: {indices_expanded.shape} (Indices expanded for head_dim)")
        print("---------------------------------------------------------")

    # Gather the sparse K and V tensors and make them contiguous
    k_sparse = torch.gather(k_expanded, 3, indices_expanded).contiguous()
    v_sparse = torch.gather(v_expanded, 3, indices_expanded).contiguous()

    if decode_step == 0:
        print(f"k_sparse.shape:         {k_sparse.shape} (Gathered top-k keys)")
        print(f"v_sparse.shape:         {v_sparse.shape} (Gathered top-k values)")
        print("---------------------------------------------------------")
    
    # --- Timed Calculation Block ---
    torch.cuda.synchronize()
    start_calc_time = time.time()
    
    # Unsqueeze q for batch matrix multiplication
    q_unsqueezed = q.unsqueeze(3)
    
    sparse_scores = torch.matmul(q_unsqueezed, k_sparse.transpose(-2, -1)).squeeze(3) / math.sqrt(hs)
    sparse_attn_weights = torch.nn.functional.softmax(sparse_scores, dim=-1)
    
    # Unsqueeze weights for batch matrix multiplication
    weights_unsqueezed = sparse_attn_weights.unsqueeze(3)
    output = torch.matmul(weights_unsqueezed, v_sparse).squeeze(3)
    
    torch.cuda.synchronize()
    end_calc_time = time.time()
    calculation_time = end_calc_time - start_calc_time
    # --- End Timed Block ---

    if decode_step == 0:
        print(f"q_unsqueezed.shape:     {q_unsqueezed.shape} (For MatMul with k_sparse)")
        print(f"sparse_scores.shape:    {sparse_scores.shape} (Final scores for sparse attention)")
        print(f"sparse_attn_weights.shape: {sparse_attn_weights.shape} (Softmax over sparse scores)")
        print(f"output.shape:           {output.shape} (Final output tensor)")
        print("--- End of Shape Inspection ---")

    return output, top_k_indices, calculation_time

if __name__ == '__main__':

    batch_size = 1
    num_heads = 32
    seq_length = 4096
    head_dim = 128
    dtype = torch.bfloat16
    device = 'cuda'

    q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device, dtype=dtype)
    
    # 也可以测试GQA
    # num_kv_heads = 8 
    # k_gqa = torch.randn(batch_size, num_kv_heads, seq_length, head_dim, device=device, dtype=dtype)
    # v_gqa = torch.randn(batch_size, num_kv_heads, seq_length, head_dim, device=device, dtype=dtype)

    # 预热：第一次运行可能包含CUDA内核加载等开销，结果不准，先运行一次进行预热
    _ = _eager_attention(q, k, v)
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()

    # start_time = time.time()

    iterations = 1
    # for _ in tqdm(range(iterations)):
    #     output = _eager_attention(q, k, v)

    # 等待所有CUDA核心完成工作
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    
    # end_time = time.time()

    # total_time = end_time - start_time
    # avg_time_per_iteration = total_time / iterations

    # print("\n--- 测试结果 ---")
    # print(f"输出张量的形状: {output.shape}")
    # print(f"总共运行次数: {iterations}")
    # print(f"总耗时: {total_time:.4f} 秒")
    # print(f"平均每次调用耗时: {avg_time_per_iteration * 1000:.4f} 毫秒")
    
    print("--- 2. 测试 ReQuest Attention ---")
    
    k_val = int(seq_length * 0.1)
    topk_update_interval = 16

    cached_indices = None
    
    _, cached_indices, _ = _request_attention(q, k, v, k_val, 0, topk_update_interval, cached_indices)
    torch.cuda.synchronize()
    
    total_internal_calculation_time = 0.0
    start_time = time.time()
    
    # for step in tqdm(range(iterations), desc="ReQuest Attention"):
    #     output_request, cached_indices, internal_time = _request_attention(
    #         q, k, v, 
    #         top_k=k_val, 
    #         decode_step=step, 
    #         topk_update_interval=topk_update_interval,
    #         cached_topk_indices=cached_indices
    #     )
    #     total_internal_calculation_time += internal_time
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time_request = end_time - start_time
    avg_time_request = total_time_request / iterations
    avg_internal_time = total_internal_calculation_time / iterations
    
    print(f"输出张量形状: {output_request.shape}")
    print(f"缓存Indices的设备: {cached_indices.device}")
    print(f"--- 函数整体耗时 ---")
    print(f"总耗时: {total_time_request:.4f} 秒")
    print(f"平均每次调用耗时: {avg_time_request * 1000:.4f} 毫秒")
    print(f"--- 内部核心计算耗时 (q, k_sparse, v_sparse -> output) ---")
    print(f"内部计算总耗时: {total_internal_calculation_time:.4f} 秒")
    print(f"平均每次内部计算耗时: {avg_internal_time * 1000:.4f} 毫秒\n")