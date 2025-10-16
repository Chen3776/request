import torch
from torch.nn.attention.flex_attention import flex_attention, BlockMask
import pdb

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# 1. 定义参数
B = 1
H = 1
Q_LEN = 8192
KV_LEN = 8192
Q_HEAD_DIM = 64
KV_HEAD_DIM = 64
device = "cuda"
dtype = torch.float16

# 2. 定义想要的 Block Size
BLOCK_SIZE_Q = 1
BLOCK_SIZE_KV = 1

# 3. 手动计算和创建 BlockMask 所需的张量
q_num_blocks_per_sequence = (Q_LEN + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
kv_num_blocks_per_sequence = (KV_LEN + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV

# q_num_blocks: 第 i 个 q_block 关注 i+1 个 kv_block -> [1, 2, 3, ...]
q_num_blocks = torch.arange(1, q_num_blocks_per_sequence + 1, device=device, dtype=torch.int32).reshape(1, 1, -1)

# q_indices: 填充每个 q_block 关注的 kv_block 的索引
max_kv_blocks_per_q = kv_num_blocks_per_sequence
q_indices = torch.zeros(B, H, q_num_blocks_per_sequence, max_kv_blocks_per_q, dtype=torch.int32, device=device)
for i in range(q_num_blocks_per_sequence):
    num_kv_to_attend = i + 1
    q_indices[:, :, i, :num_kv_to_attend] = torch.arange(num_kv_to_attend, device=device)

# kv_num_blocks: 第 j 个 kv_block 被 N-j 个 q_block 关注 -> [N, N-1, ...]
# --- 这是关键的修正 ---
kv_num_blocks = torch.arange(kv_num_blocks_per_sequence, 0, -1, device=device, dtype=torch.int32).reshape(1, 1, -1)

# kv_indices: 填充每个 kv_block 被哪些 q_block 关注
# 注意：对于 flex_attention 的前向计算，kv_indices 实际上可以是一个简单的arange，
# 因为实际的稀疏计算是由 q_indices 和 q_num_blocks 主导的。
# 我们保持之前的简单构造方式。
kv_indices = torch.arange(0, kv_num_blocks_per_sequence, device=device, dtype=torch.int32).reshape(1, 1, -1, 1).expand(B, H, kv_num_blocks_per_sequence, 1)

# full_* arguments
# 对于因果掩码，稀疏版本和完整版本逻辑相同
full_q_num_blocks = q_num_blocks.clone()
full_q_indices = q_indices.clone()
# --- 修正 full_kv_num_blocks 以匹配新的逻辑 ---
full_kv_num_blocks = kv_num_blocks.clone()
# full_kv_indices 需要列出所有 kv 块
full_kv_indices = torch.arange(0, kv_num_blocks_per_sequence, device=device, dtype=torch.int32).reshape(1, 1, -1)


# 4. 实例化 BlockMask
manual_block_mask = BlockMask(
    q_num_blocks=q_num_blocks,
    q_indices=q_indices,
    kv_num_blocks=kv_num_blocks,
    # kv_indices 的构造对于前向传播不是最关键的，但为了完整性我们传入
    # 这里我们传入一个更简单的版本，因为 q_indices 已经定义了完整的图
    kv_indices=torch.arange(0, kv_num_blocks_per_sequence, device=device, dtype=torch.int32).reshape(1, 1, -1),
    full_q_num_blocks=full_q_num_blocks,
    full_q_indices=full_q_indices,
    full_kv_num_blocks=full_kv_num_blocks,
    full_kv_indices=full_kv_indices,
    BLOCK_SIZE=(BLOCK_SIZE_Q, BLOCK_SIZE_KV),
    seq_lengths=(Q_LEN, KV_LEN),
    mask_mod=causal_mask
)

# 示例输入张量
query = torch.randn(B, H, Q_LEN, Q_HEAD_DIM, device=device, dtype=dtype)
key = torch.randn(B, H, KV_LEN, KV_HEAD_DIM, device=device, dtype=dtype)
value = torch.randn(B, H, KV_LEN, KV_HEAD_DIM, device=device, dtype=dtype)

pdb.set_trace()
output = flex_attention(query, key, value, block_mask=manual_block_mask)

print("Attention output shape:", output.shape)
print("BlockMask created successfully with BLOCK_SIZE:", manual_block_mask.BLOCK_SIZE)