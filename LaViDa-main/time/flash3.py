import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import triton
import triton.language as tl
import math
from typing import Optional

# ==============================================================================
# KERNEL 1: Top-K Indices Selection
# ==============================================================================

@triton.jit
def _topk_indices_kernel(
    Q, K, O_indices,
    stride_q_bs, stride_q_h, stride_q_t, stride_q_d,
    stride_k_bs, stride_k_h, stride_k_t, stride_k_d,
    stride_oi_bs, stride_oi_h, stride_oi_t, stride_oi_k,
    T_k: tl.int32,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_query = tl.program_id(1)

    q_offset = pid_batch * stride_q_h + pid_query * stride_q_t
    q_ptr = Q + q_offset
    k_offset = pid_batch * stride_k_h
    k_ptr = K + k_offset

    q = tl.load(q_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    q = q * (HEAD_DIM ** -0.5)

    top_k_scores = tl.full((TOP_K,), -float('inf'), dtype=tl.float32)
    top_k_indices = tl.full((TOP_K,), -1, dtype=tl.int32)
    
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, T_k, BLOCK_SIZE_N):
        k_block_ptr = k_ptr + (start_n + offs_n)[:, None] * stride_k_t + tl.arange(0, HEAD_DIM)[None, :]
        mask = (start_n + offs_n) < T_k
        k_block = tl.load(k_block_ptr, mask=mask[:, None], other=0.0)
        
        scores = tl.sum(q[None, :] * k_block, axis=1)
        scores = tl.where(mask, scores, -float('inf'))

        for i in range(BLOCK_SIZE_N):
            current_score = scores[i]
            current_index = start_n + offs_n[i]

            # --- 核心修改在这里 ---
            # 1. 使用 tl.argmin 获取最小值的索引 (用于 tl.where)
            min_idx = tl.argmin(top_k_scores, axis=0)
            # 2. 使用 tl.min 获取最小值的数值 (用于 if 判断)
            min_score = tl.min(top_k_scores, axis=0)

            # 如果新分数更大，就替换掉最小值
            if current_score > min_score:
                top_k_scores = tl.where(tl.arange(0, TOP_K) == min_idx, current_score, top_k_scores)
                top_k_indices = tl.where(tl.arange(0, TOP_K) == min_idx, current_index, top_k_indices)

    # --- 在所有 Key 遍历完后，进行一次最终排序 ---
    # (这部分的选择排序虽然效率不高，但对于编译是正确的)
    for i in range(TOP_K):
        for j in range(i + 1, TOP_K):
            val1_score = top_k_scores[i]
            val2_score = top_k_scores[j]
            val1_idx = top_k_indices[i]
            val2_idx = top_k_indices[j]
            
            swap_mask = val1_score < val2_score
            
            new_val1_score = tl.where(swap_mask, val2_score, val1_score)
            new_val2_score = tl.where(swap_mask, val1_score, val2_score)
            new_val1_idx = tl.where(swap_mask, val2_idx, val1_idx)
            new_val2_idx = tl.where(swap_mask, val1_idx, val2_idx)

            top_k_scores = tl.where(tl.arange(0, TOP_K) == i, new_val1_score, top_k_scores)
            top_k_scores = tl.where(tl.arange(0, TOP_K) == j, new_val2_score, top_k_scores)
            top_k_indices = tl.where(tl.arange(0, TOP_K) == i, new_val1_idx, top_k_indices)
            top_k_indices = tl.where(tl.arange(0, TOP_K) == j, new_val2_idx, top_k_indices)

    output_ptr = O_indices + pid_batch * stride_oi_h + pid_query * stride_oi_t
    tl.store(output_ptr + tl.arange(0, TOP_K), top_k_indices)

# ==============================================================================
# KERNEL 2: Sparse Attention Calculation
# ==============================================================================

@triton.jit
def _sparse_attention_kernel(
    Q, K, V, O, top_k_indices,
    stride_q_bs, stride_q_h, stride_q_t, stride_q_d,
    stride_k_bs, stride_k_h, stride_k_t, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_t, stride_v_d,
    stride_o_bs, stride_o_h, stride_o_t, stride_o_d,
    stride_i_bs, stride_i_h, stride_i_t, stride_i_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_SPARSE: tl.constexpr,
    TOP_K: tl.constexpr,
):
    start_m = tl.program_id(0)
    pid_batch_head = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_ptr = Q + pid_batch_head * stride_q_h + offs_m[:, None] * stride_q_t + offs_d[None, :]
    k_ptr_base = K + pid_batch_head * stride_k_h
    v_ptr_base = V + pid_batch_head * stride_v_h
    indices_ptr = top_k_indices + pid_batch_head * stride_i_h + offs_m[:, None] * stride_i_t
    o_ptr = O + pid_batch_head * stride_o_h + offs_m[:, None] * stride_o_t + offs_d[None, :]

    q = tl.load(q_ptr)
    
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    sm_scale = HEAD_DIM ** -0.5

    for start_k in range(0, TOP_K, BLOCK_K_SPARSE):
        offs_k = start_k + tl.arange(0, BLOCK_K_SPARSE)
        
        mask_2d = offs_k[None, :] < TOP_K
        
        k_indices = tl.load(indices_ptr + offs_k, mask=mask_2d, other=0)
        
        k_ptr = k_ptr_base + k_indices[:, :, None] * stride_k_t + offs_d[None, None, :]
        v_ptr = v_ptr_base + k_indices[:, :, None] * stride_v_t + offs_d[None, None, :]
        
        mask_3d_for_load = mask_2d[:, :, None]
        k = tl.load(k_ptr, mask=mask_3d_for_load, other=0.0)
        v = tl.load(v_ptr, mask=mask_3d_for_load, other=0.0)

        q_expanded = q[:, None, :]
        s = tl.sum(q_expanded * k, axis=2) * sm_scale
        
        s = tl.where(mask_2d, s, -float('inf'))
        
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        p_expanded = p[:, :, None]
        acc = acc * alpha[:, None]
        acc += tl.sum(p_expanded * v, axis=1)
        
        m_i = m_new
        
    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Q.dtype.element_ty))

# ==============================================================================
# Python Wrapper Function
# ==============================================================================

def request_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    cached_indices: Optional[torch.Tensor] = None
) -> (torch.Tensor, torch.Tensor):

    B, nh, T_q, hs = q.shape
    T_k = k.shape[2]
    effective_top_k = min(top_k, T_k)
    
    if effective_top_k <= 0:
        return torch.zeros_like(q), None

    output = torch.empty_like(q)
    
    if cached_indices is None:
        
        padded_top_k = triton.next_power_of_2(effective_top_k)
        
        top_k_indices_padded = torch.empty((B, nh, T_q, padded_top_k), dtype=torch.int32, device=q.device)
        
        grid_topk = (B * nh, T_q)
        
        _topk_indices_kernel[grid_topk](
            q, k, top_k_indices_padded,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            top_k_indices_padded.stride(0), top_k_indices_padded.stride(1), top_k_indices_padded.stride(2), top_k_indices_padded.stride(3),
            T_k,
            HEAD_DIM=hs,
            BLOCK_SIZE_N=64,
            TOP_K=padded_top_k,
        )
        
        top_k_indices = top_k_indices_padded[:, :, :, :effective_top_k]
        
    else:
        top_k_indices = cached_indices

    grid_attn = (triton.cdiv(T_q, 16), B * nh)
    
    _sparse_attention_kernel[grid_attn](
        q, k, v, output, top_k_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        top_k_indices.stride(0), top_k_indices.stride(1), top_k_indices.stride(2), top_k_indices.stride(3),
        HEAD_DIM=hs,
        BLOCK_M=16,
        BLOCK_K_SPARSE=64,
        TOP_K=effective_top_k,
        num_warps=4,
        num_stages=2,
    )
    
    return output, top_k_indices

def test_request_attention_staged(BS, HEAD, SEQLEN, DIM, TOP_K):

    print(f"--- Staged Test with BS={BS}, HEAD={HEAD}, SEQLEN={SEQLEN}, DIM={DIM}, TOP_K={TOP_K} ---")
    dtype = torch.float16

    # 1. 准备输入数据
    q = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    
    # 准备一个随机的、合法的 top_k_indices 用于第一阶段测试
    # torch.randint生成的是[low, high)的整数
    random_indices = torch.randint(0, SEQLEN, (BS, HEAD, SEQLEN, TOP_K), device="cuda", dtype=torch.int32)

    # 2. 统一预热
    print("Warming up kernels...")
    _, _ = request_attention_triton(q, k, v, top_k=TOP_K, cached_indices=random_indices)
    _, _ = request_attention_triton(q, k, v, top_k=TOP_K, cached_indices=None)
    torch.cuda.synchronize()
    print("Warm-up complete. Starting timed runs.")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("\n--- Stage 1: Testing Sparse Attention Kernel (Indices Provided) ---")
    start_event.record()
    _ = request_attention_triton(q, k, v, top_k=TOP_K, cached_indices=random_indices)
    end_event.record()
    torch.cuda.synchronize()
    time_sparse_only = start_event.elapsed_time(end_event)
    print(f"Time (Sparse Attention only): {time_sparse_only:.4f} ms")

    print("\n--- Stage 2: Testing End-to-End (Indices Generated + Sparse Attention) ---")
    start_event.record()
    tri_out_e2e, _ = request_attention_triton(q, k, v, top_k=TOP_K, cached_indices=None)
    end_event.record()
    torch.cuda.synchronize()
    time_end_to_end = start_event.elapsed_time(end_event)
    print(f"Time (End-to-End): {time_end_to_end:.4f} ms")
    
    time_indexing = time_end_to_end - time_sparse_only
    print(f"Estimated time for Index Generation: {time_indexing:.4f} ms")

def test_attention():
    BS, HEAD, SEQLEN, DIM = 1, 32, 8192, 128
    TOP_K = SEQLEN // 10
    
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    test_request_attention_staged(BS, HEAD, SEQLEN, DIM, TOP_K)
    # causal_test(BS, HEAD, SEQLEN, DIM, causal=False)
    # causal_test(BS, HEAD, SEQLEN, DIM, causal=True)
    
if __name__ == '__main__':
    test_attention()