import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pdb
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)

BATCH_SIZE = 1
NUM_HEADS = 32
SEQ_LEN_Q = 128
SEQ_LEN_KV = 128
HEAD_DIM = 128
BLOCK_SIZE = 2
DTYPE = torch.bfloat16
device = 'cuda'

num_blocks_q = SEQ_LEN_Q // BLOCK_SIZE
num_blocks_kv = SEQ_LEN_KV // BLOCK_SIZE
keep_ratio = 0.1

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, 1, 1, SEQ_LEN_Q, SEQ_LEN_KV, device="cuda", BLOCK_SIZE=BLOCK_SIZE, _compile=False)
q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM, dtype=DTYPE, device=device)
k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=DTYPE, device=device)
v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM, dtype=DTYPE, device=device)

output = flex_attention(q, k, v, block_mask=block_mask)
print(output.shape)