import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import triton
import triton.language as tl

# ==============================================================================
# KERNEL 1: (新增) 为每个Query块计算一个“代表性”Query向量 (取平均值)
# ==============================================================================
@triton.jit
def _compute_query_representatives_kernel(
    Q, Rep_Q,
    stride_q_bs, stride_q_h, stride_q_t, stride_q_d,
    stride_rq_bs, stride_rq_h, stride_rq_t, stride_rq_d,
    T_q: tl.int32, HEAD_DIM: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 计算一个 Query 块的代表
    pid_q_block = tl.program_id(0)
    pid_batch_head = tl.program_id(1)

    # 计算当前Query块的起始行
    start_m = pid_q_block * QUERY_BLOCK_SIZE
    offs_m = start_m + tl.arange(0, QUERY_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    
    mask_q = offs_m[:, None] < T_q
    
    q_ptr = Q + pid_batch_head * stride_q_h + offs_m[:, None] * stride_q_t + offs_d[None, :]
    q_block = tl.load(q_ptr, mask=mask_q, other=0.0)

    # 计算块内平均值作为代表
    # tl.sum的返回值类型跟随输入，所以先转为float32保证精度
    rep_q = tl.sum(q_block.to(tl.float32), axis=0) / QUERY_BLOCK_SIZE
    
    # 存储代表性Query
    rq_ptr = Rep_Q + pid_batch_head * stride_rq_h + pid_q_block * stride_rq_t + offs_d
    tl.store(rq_ptr, rep_q.to(Rep_Q.dtype.element_ty))

# ==============================================================================
# KERNEL 2: (修改) 使用“代表性Query”为每个Query块寻找Top-K的Key块
# ==============================================================================
@triton.jit
def _topk_block_scores_kernel(
    Rep_Q, K, O_block_scores,
    stride_rq_bs, stride_rq_h, stride_rq_t, stride_rq_d,
    stride_k_bs, stride_k_h, stride_k_t, stride_k_d,
    stride_obs_bs, stride_obs_h, stride_obs_t, stride_obs_k,
    num_q_blocks: tl.int32, T_k: tl.int32, HEAD_DIM: tl.constexpr,
    REP_Q_BLOCK_SIZE: tl.constexpr, # 新增：定义一次处理多少个Rep Q
    BLOCK_K_SIZE: tl.constexpr,
):
    # 每个 program 计算一个 (代表性Query块, Key块) 的交互
    pid_rq_block = tl.program_id(0)
    pid_k_block = tl.program_id(1)
    pid_batch_head = tl.program_id(2)

    # --- 加载代表性Query块 ---
    start_rq_row = pid_rq_block * REP_Q_BLOCK_SIZE
    offs_m = start_rq_row + tl.arange(0, REP_Q_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    
    mask_rq = offs_m[:, None] < num_q_blocks
    
    rq_ptr = Rep_Q + pid_batch_head * stride_rq_h + offs_m[:, None] * stride_rq_t + offs_d[None, :]
    rep_q_block = tl.load(rq_ptr, mask=mask_rq, other=0.0) # shape: (REP_Q_BLOCK_SIZE, D)

    # --- 加载Key块 ---
    start_k_col = pid_k_block * BLOCK_K_SIZE
    offs_k = start_k_col + tl.arange(0, BLOCK_K_SIZE)
    mask_k = offs_k < T_k
    
    k_ptr = K + pid_batch_head * stride_k_h + offs_k[None, :] * stride_k_t + offs_d[:, None]
    k_block = tl.load(k_ptr, mask=mask_k[None, :], other=0.0) # shape: (D, BLOCK_K_SIZE)
    
    # --- 计算分数 ---
    # 现在是 (M, K) @ (K, N)，其中 M=REP_Q_BLOCK_SIZE，满足 M >= 16
    scores = tl.dot(rep_q_block, k_block.to(rep_q_block.dtype)) # shape: (M, N)
    sm_scale = HEAD_DIM ** -0.5
    scores *= sm_scale
    
    # 应用掩码
    full_mask = mask_rq & mask_k[None, :]
    scores = tl.where(full_mask, scores, -float('inf'))
    
    # 计算每个Rep Q对这个Key块的最大分数
    block_max_scores = tl.max(scores, axis=1) # shape: (M,)
    
    # --- 存储结果 ---
    output_ptr = O_block_scores + pid_batch_head * stride_obs_h + offs_m * stride_obs_t + pid_k_block
    tl.store(output_ptr, block_max_scores, mask=mask_rq[:, 0])

# ==============================================================================
# KERNEL 3: (重写) 高性能的块稀疏注意力计算
# ==============================================================================
def quest_attention_triton(q, k, v, top_k_blocks, query_block_size, key_block_size):
    B, nh, T_q, hs = q.shape
    T_k = k.shape[2]
    
    assert T_q % query_block_size == 0, "SeqLen Q must be divisible by Q block size"
    assert T_k % key_block_size == 0, "SeqLen K must be divisible by K block size"
    num_q_blocks = T_q // query_block_size
    num_k_blocks = T_k // key_block_size
    
    effective_top_k_blocks = min(top_k_blocks, num_k_blocks)
    if effective_top_k_blocks <= 0: return torch.zeros_like(q), None
    
    # 阶段 1: 计算代表性Query (不变)
    rep_q = torch.empty((B, nh, num_q_blocks, hs), dtype=torch.float32, device=q.device)
    grid_rep_q = (num_q_blocks, B * nh)
    _compute_query_representatives_kernel[grid_rep_q](
        q, rep_q,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        rep_q.stride(0), rep_q.stride(1), rep_q.stride(2), rep_q.stride(3),
        T_q, HEAD_DIM=hs, QUERY_BLOCK_SIZE=query_block_size
    )
    
    # 阶段 2: 用代表性Query块选择Top-K的Key块
    block_scores = torch.empty((B, nh, num_q_blocks, num_k_blocks), dtype=torch.float32, device=q.device)
    
    # ================= 唯一的修改点在这里 =================
    REP_Q_BLOCK_SIZE = 16 # 定义一次处理16个Rep Q，满足 M >= 16 的要求
    grid_scores = (triton.cdiv(num_q_blocks, REP_Q_BLOCK_SIZE), num_k_blocks, B * nh)
    _topk_block_scores_kernel[grid_scores](
        rep_q, k, block_scores,
        rep_q.stride(0), rep_q.stride(1), rep_q.stride(2), rep_q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        block_scores.stride(0), block_scores.stride(1), block_scores.stride(2), block_scores.stride(3),
        num_q_blocks, T_k, HEAD_DIM=hs, 
        REP_Q_BLOCK_SIZE=REP_Q_BLOCK_SIZE, 
        BLOCK_K_SIZE=key_block_size
    )
    # ====================================================

    _, top_k_block_indices = torch.topk(block_scores, k=effective_top_k_blocks, dim=-1, sorted=False)
    top_k_block_indices = top_k_block_indices.to(torch.int32)

    # 阶段 3: 调用最终的高性能块稀疏计算Kernel (不变)
    output = torch.empty_like(q)
    grid_attn = (num_q_blocks, B * nh)
    _quest_attention_kernel[grid_attn](
        q, k, v, output, top_k_block_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        top_k_block_indices.stride(0), top_k_block_indices.stride(1), top_k_block_indices.stride(2), top_k_block_indices.stride(3),
        T_q, T_k, HEAD_DIM=hs,
        QUERY_BLOCK_SIZE=query_block_size, KEY_BLOCK_SIZE=key_block_size,
        TOP_K_BLOCKS=effective_top_k_blocks,
        num_warps=4, num_stages=3
    )
    return output, top_k_block_indices

# ==============================================================================
# Python Wrapper Function
# ==============================================================================
def quest_attention_triton(q, k, v, top_k_blocks, query_block_size, key_block_size):
    B, nh, T_q, hs = q.shape
    T_k = k.shape[2]
    
    assert T_q % query_block_size == 0, "SeqLen Q must be divisible by Q block size"
    assert T_k % key_block_size == 0, "SeqLen K must be divisible by K block size"
    num_q_blocks = T_q // query_block_size
    num_k_blocks = T_k // key_block_size
    
    effective_top_k_blocks = min(top_k_blocks, num_k_blocks)
    if effective_top_k_blocks <= 0: return torch.zeros_like(q), None
    
    # 阶段 1: 计算代表性Query
    rep_q = torch.empty((B, nh, num_q_blocks, hs), dtype=torch.float32, device=q.device)
    grid_rep_q = (num_q_blocks, B * nh)
    _compute_query_representatives_kernel[grid_rep_q](
        q, rep_q,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        rep_q.stride(0), rep_q.stride(1), rep_q.stride(2), rep_q.stride(3),
        T_q, HEAD_DIM=hs, QUERY_BLOCK_SIZE=query_block_size
    )
    
    # 阶段 2: 用代表性Query块选择Top-K的Key块
    block_scores = torch.empty((B, nh, num_q_blocks, num_k_blocks), dtype=torch.float32, device=q.device)
    
    REP_Q_BLOCK_SIZE = 16 
    grid_scores = (triton.cdiv(num_q_blocks, REP_Q_BLOCK_SIZE), num_k_blocks, B * nh)
    
    # ================= 唯一的修改点在这里 =================
    # 在 T_k 前面补上了缺失的 num_q_blocks 参数
    _topk_block_scores_kernel[grid_scores](
        rep_q, k, block_scores,
        rep_q.stride(0), rep_q.stride(1), rep_q.stride(2), rep_q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        block_scores.stride(0), block_scores.stride(1), block_scores.stride(2), block_scores.stride(3),
        num_q_blocks,  # <--- 补上这个缺失的参数
        T_k,
        HEAD_DIM=hs, 
        REP_Q_BLOCK_SIZE=REP_Q_BLOCK_SIZE, 
        BLOCK_K_SIZE=key_block_size
    )
    # ====================================================

    _, top_k_block_indices = torch.topk(block_scores, k=effective_top_k_blocks, dim=-1, sorted=False)
    top_k_block_indices = top_k_block_indices.to(torch.int32)

    # 阶段 3: 调用最终的高性能块稀疏计算Kernel
    output = torch.empty_like(q)
    grid_attn = (num_q_blocks, B * nh)
    _quest_attention_kernel[grid_attn](
        q, k, v, output, top_k_block_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        top_k_block_indices.stride(0), top_k_block_indices.stride(1), top_k_block_indices.stride(2), top_k_block_indices.stride(3),
        T_q, T_k, HEAD_DIM=hs,
        QUERY_BLOCK_SIZE=query_block_size, KEY_BLOCK_SIZE=key_block_size,
        TOP_K_BLOCKS=effective_top_k_blocks,
        num_warps=4, num_stages=3
    )
    return output, top_k_block_indices

def test_quest_attention():
    BS, HEAD, SEQLEN, DIM = 1, 32, 8192, 128
    QUERY_BLOCK_SIZE = 64
    KEY_BLOCK_SIZE = 128
    TOP_K_BLOCKS = (SEQLEN // KEY_BLOCK_SIZE) // 10

    print(f"--- Quest-like Block-Sparse Test ---")
    print(f"Parameters: Q_BLOCK={QUERY_BLOCK_SIZE}, K_BLOCK={KEY_BLOCK_SIZE}, TOP_K_BLOCKS={TOP_K_BLOCKS}")
    
    dtype = torch.float16
    torch.manual_seed(2025)
    
    q = torch.randn((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda")
    k = torch.randn((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda")
    v = torch.randn((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda")
    
    print("Warming up kernels...")
    if TOP_K_BLOCKS > 0:
        _, _ = quest_attention_triton(q, k, v, TOP_K_BLOCKS, QUERY_BLOCK_SIZE, KEY_BLOCK_SIZE)
        torch.cuda.synchronize()
    print("Warm-up complete.")

    if TOP_K_BLOCKS > 0:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        tri_out, _ = quest_attention_triton(q, k, v, TOP_K_BLOCKS, QUERY_BLOCK_SIZE, KEY_BLOCK_SIZE)
        end_event.record()
        torch.cuda.synchronize()
        
        time_quest = start_event.elapsed_time(end_event)
        print(f"\nTime (End-to-End Quest Attention): {time_quest:.4f} ms")
    else:
        print("\nSkipping timing, TOP_K_BLOCKS is 0.")

if __name__ == '__main__':
    test_quest_attention()