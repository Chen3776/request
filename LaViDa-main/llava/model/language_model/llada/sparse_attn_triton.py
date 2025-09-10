# ===== sparse_attn_triton.py =====
import torch
import triton
import triton.language as tl

@triton.jit
def _sparse_attention_fwd_kernel(
    Q, K, V, TopKIdx,
    Out,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_idx_b, stride_idx_h, stride_idx_t, stride_idx_k,
    stride_ob, stride_oh, stride_ot, stride_od,
    T_q, T_k, K_VAL, HS,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_pos = tl.program_id(2)

    # 在 head_dim 上分块循环处理
    d_start = 0
    for d_offset in range(0, HS, BLOCK_D):
        d_range = d_offset + tl.arange(0, BLOCK_D)
        d_mask = d_range < HS

        # 加载当前分块的 Q 向量
        q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh + q_pos * stride_qt + d_range * stride_qd
        q_vals = tl.load(q_ptr, mask=d_mask, other=0.0)  # (BLOCK_D,)

        # 初始化累加器
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        l_i = 0.0
        m_i = -float('inf')

        # 遍历 Top-K 分块
        for k_block in range(0, K_VAL, BLOCK_K):
            # 创建当前块的索引 [k_block, k_block+1, ..., k_block+BLOCK_K-1]
            i_range = tl.arange(0, BLOCK_K) + k_block
            mask_k = i_range < K_VAL

            # 从 TopKIdx 加载对应的 key 位置 (BLOCK_K,)
            k_pos = tl.load(
                TopKIdx + batch_idx * stride_idx_b + head_idx * stride_idx_h + q_pos * stride_idx_t + (i_range % K_VAL) * stride_idx_k,
                mask=mask_k,
                other=0
            )

            # 构造 K 指针 (BLOCK_K, BLOCK_D)
            k_ptrs = (
                K 
                + batch_idx * stride_kb 
                + head_idx * stride_kh 
                + k_pos[:, None] * stride_kt 
                + d_range[None, :] * stride_kd
            )
            k_vals = tl.load(k_ptrs, mask=mask_k[:, None] & d_mask[None, :], other=0.0)

            # 计算 QK 分数 (BLOCK_K,)
            qk = tl.sum(q_vals[None, :] * k_vals, axis=1) * SCALE

            # 稳定 softmax（保持不变）
            m_ij = tl.maximum(m_i, tl.max(qk))
            p = tl.exp(qk - m_ij)
            l_ij = tl.sum(p)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        # 重新加载 V 并加权求和（简化版）
        for k_block in range(0, K_VAL, BLOCK_K):
            i_range = tl.arange(0, BLOCK_K) + k_block
            mask_k = i_range < K_VAL

            k_pos = tl.load(
                TopKIdx + batch_idx * stride_idx_b + head_idx * stride_idx_h + q_pos * stride_idx_t + (i_range % K_VAL) * stride_idx_k,
                mask=mask_k,
                other=0
            )

            # 加载 V
            v_ptrs = (
                V 
                + batch_idx * stride_vb 
                + head_idx * stride_vh 
                + k_pos[:, None] * stride_vt 
                + d_range[None, :] * stride_vd
            )
            v_vals = tl.load(v_ptrs, mask=mask_k[:, None] & d_mask[None, :], other=0.0)

            # 重新加载 K 计算权重（可优化）
            k_ptrs = (
                K 
                + batch_idx * stride_kb 
                + head_idx * stride_kh 
                + k_pos[:, None] * stride_kt 
                + d_range[None, :] * stride_kd
            )
            k_vals = tl.load(k_ptrs, mask=mask_k[:, None] & d_mask[None, :], other=0.0)
            qk = tl.sum(q_vals[None, :] * k_vals, axis=1) * SCALE
            p = tl.exp(qk - m_i) / l_i

            # 加权求和
            acc += tl.sum(p[:, None] * v_vals, axis=0)

        # 存储当前分块输出
        out_ptr = Out + batch_idx * stride_ob + head_idx * stride_oh + q_pos * stride_ot + d_range * stride_od
        tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=d_mask)


def sparse_attention_triton(q, k, v, topk_indices, hs, dropout_p=0.0, training=False):
    """
    使用 Triton 实现稀疏注意力。
    输入：
        q: (B, nH, T_q, hs)
        k: (B, nH, T_k, hs)
        v: (B, nH, T_k, hs)
        topk_indices: (B, nH, T_q, K_VAL)
    输出：
        output: (B, nH, T_q, hs)
    """
    assert q.shape[-1] == hs, f"Expected head size {hs}, but got {q.shape[-1]}"
    assert q.shape[:2] == v.shape[:2], f"Batch/Head shape mismatch: q{q.shape[:2]} vs v{v.shape[:2]}"
    assert topk_indices.shape[-1] > 0, "K must be > 0"

    B, nH, T_q, _ = q.shape
    K_VAL = topk_indices.shape[-1]
    SCALE = hs ** -0.5

    # 输出张量
    output = torch.empty_like(q)

    # 配置并行维度
    grid = (B, nH, T_q)
    BLOCK_D = 64  # 每个程序处理 64 维（可调）
    BLOCK_K = 32  # 每次处理 32 个 Top-K 索引（可调）

    # 启动 Triton kernel
    _sparse_attention_fwd_kernel[grid](
        q, k, v, topk_indices, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        topk_indices.stride(0), topk_indices.stride(1), topk_indices.stride(2), topk_indices.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        T_q, k.shape[2], K_VAL, hs,
        SCALE=SCALE,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    # 如果训练且需要 dropout
    if training and dropout_p > 0.0:
        output = torch.nn.functional.dropout(output, p=dropout_p)

    return output