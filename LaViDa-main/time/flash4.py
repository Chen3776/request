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
def _full_scores_kernel(
    Q, K, O_scores,
    stride_q_bs, stride_q_h, stride_q_t, stride_q_d,
    stride_k_bs, stride_k_h, stride_k_t, stride_k_d,
    stride_os_bs, stride_os_h, stride_os_t, stride_os_k,
    T_q: tl.int32, T_k: tl.int32,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch_head = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptr = Q + pid_batch_head * stride_q_h + offs_m[:, None] * stride_q_t + offs_d[None, :]
    k_ptr = K + pid_batch_head * stride_k_h + offs_n[None, :] * stride_k_t + offs_d[:, None]
    
    mask_m = offs_m[:, None] < T_q
    mask_n = offs_n[None, :] < T_k
    
    q = tl.load(q_ptr, mask=mask_m, other=0.0)
    k = tl.load(k_ptr, mask=mask_n, other=0.0)
    
    acc = tl.dot(q, k)
    
    sm_scale = HEAD_DIM ** -0.5
    acc = acc * sm_scale

    output_ptr = O_scores + pid_batch_head * stride_os_h + offs_m[:, None] * stride_os_t + offs_n[None, :]
    output_mask = mask_m & mask_n
    tl.store(output_ptr, acc, mask=output_mask)

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
        # 步骤 1: 使用 Triton Kernel 计算全量分数
        full_scores = torch.empty((B, nh, T_q, T_k), dtype=torch.float32, device=q.device)
        
        grid_scores = (triton.cdiv(T_q, 32), triton.cdiv(T_k, 32), B * nh)
        
        _full_scores_kernel[grid_scores](
            q, k, full_scores,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            full_scores.stride(0), full_scores.stride(1), full_scores.stride(2), full_scores.stride(3),
            T_q, T_k,
            HEAD_DIM=hs,
            BLOCK_M=32, BLOCK_N=32,
        )
        
        # 步骤 2: 使用 PyTorch 高性能的 topk 函数来选择索引
        _, top_k_indices = torch.topk(full_scores, k=effective_top_k, dim=-1, sorted=False)
        # topk 返回 int64, 需要转换为 int32
        top_k_indices = top_k_indices.to(torch.int32)

    else:
        top_k_indices = cached_indices
    
    # 步骤 3: 调用我们已经写好的、稳定的稀疏计算 Kernel
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

def profile_materialization_vs_flash(BS, HEAD, SEQLEN, DIM):
    """
    专项测试: 对比“物化全量分数矩阵”与“Flash Attention”的耗时
    """
    print("--- Profiling: Materialization vs. Fused Flash Attention ---")
    print(f"Parameters: BS={BS}, HEAD={HEAD}, SEQLEN={SEQLEN}, DIM={DIM}")
    dtype = torch.float16
    q = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    
    full_scores_output = torch.empty((BS, HEAD, SEQLEN, SEQLEN), dtype=torch.float32, device=q.device)
    
    print("Warming up kernels...")
    grid_scores = (triton.cdiv(SEQLEN, 32), triton.cdiv(SEQLEN, 32), BS * HEAD)
    _full_scores_kernel[grid_scores](
        q, k, full_scores_output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        full_scores_output.stride(0), full_scores_output.stride(1), full_scores_output.stride(2), full_scores_output.stride(3),
        SEQLEN, SEQLEN, HEAD_DIM=DIM, BLOCK_M=32, BLOCK_N=32,
    )
    torch.cuda.synchronize()
    print("Warm-up complete. Starting timed runs.")

    # 准备计时工具
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 3. 计时: 计算完整Attn分数并写入HBM
    start_event.record()
    _full_scores_kernel[grid_scores](
        q, k, full_scores_output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        full_scores_output.stride(0), full_scores_output.stride(1), full_scores_output.stride(2), full_scores_output.stride(3),
        SEQLEN, SEQLEN, HEAD_DIM=DIM, BLOCK_M=32, BLOCK_N=32,
    )
    end_event.record()
    torch.cuda.synchronize()
    time_materialize = start_event.elapsed_time(end_event)
    
    # 5. 打印结果
    print("\n--- Profiling Results ---")
    print(f"Time to Compute & Write Full Scores (Materialization): {time_materialize:.4f} ms")


def test_attention():
    BS, HEAD, SEQLEN, DIM = 1, 32, 65536, 128
    TOP_K = SEQLEN // 20
    
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # test_request_attention_staged(BS, HEAD, SEQLEN, DIM, TOP_K)
    # profile_materialization_vs_flash(BS, HEAD, SEQLEN, DIM)
    
    # causal_test(BS, HEAD, SEQLEN, DIM, causal=False)
    # causal_test(BS, HEAD, SEQLEN, DIM, causal=True)
    
if __name__ == '__main__':
    test_attention()