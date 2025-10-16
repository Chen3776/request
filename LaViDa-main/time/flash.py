import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TORCH_USE_CUDA_DSA"]="1"

import torch
import triton
import triton.language as tl
import pdb

@triton.jit
def _flash_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    sm_scale,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_offset = pid_z * stride_qz + pid_h * stride_qh + (start_m * BLOCK_M) * stride_qm
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh + (start_m * BLOCK_M) * stride_om

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    O_ptr = O + o_offset

    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD
    pdb.set_trace()
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None])

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        
        k_ptrs = K_ptr + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :])
        
        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        
        s = tl.dot(q, k) * sm_scale
        
        m_ij = tl.max(s, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = alpha * l_i + tl.sum(p, 1)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention(q, k, v):
    q, k, v = [x.contiguous() for x in (q, k, v)]
    Z, H, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)
    
    sm_scale = 1.0 / (D_HEAD ** 0.5)

    grid = (triton.cdiv(N_CTX, 128), Z, H)

    _flash_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        sm_scale,
        D_HEAD=D_HEAD,
        BLOCK_M=128,
        BLOCK_N=64,
    )
    
    return o


if __name__ == '__main__':
    BATCH, N_HEADS, N_CTX, D_HEAD = 1, 32, 4096, 64
    
    q = torch.randn((BATCH, N_HEADS, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
    k = torch.randn((BATCH, N_HEADS, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
    v = torch.randn((BATCH, N_HEADS, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")

    print("Running PyTorch SDPA...")
    torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    print("Running Triton FlashAttention...")
    triton_output = flash_attention(q, k, v)
    print(f"Triton output is close to torch output: {torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)}")

    @triton.testing.perf_report(
        [
            triton.testing.Benchmark(
                x_names=['N_CTX'], x_vals=[128 * i for i in range(2, 33)], line_arg='provider',
                line_vals=['pytorch', 'triton'], line_names=['PyTorch', 'Triton'],
                styles=[('blue', '-'), ('green', '-')], ylabel='ms', plot_name='flash-attention-performance',
                args={'H': N_HEADS, 'D_HEAD': D_HEAD, 'BATCH': BATCH},
            )
        ]
    )
    def benchmark(N_CTX, H, D_HEAD, BATCH, provider):
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda")
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attention(q, k, v), quantiles=quantiles)
        return ms, min_ms, max_ms
    
    print("\nRunning benchmark...")
    benchmark.run(show_plots=False, print_data=True)