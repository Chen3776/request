import os
import torch
import time
from contextlib import contextmanager

# Ensure we're using a visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Import necessary components from your provided script and the new config file
from modeling_llada import LLaDASequentialBlock, BufferCache
from configuration_llada import ModelConfig

# ==============================================================================
# Helper Function to Measure Execution Time
# ==============================================================================

def measure_time(func, *args, **kwargs):
    """
    Measures the execution time of a function on CUDA.
    """
    # Warm-up runs to ensure CUDA kernels are compiled and ready
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    # Actual measurement runs
    for _ in range(100):
        func(*args, **kwargs)
    
    end_event.record()
    torch.cuda.synchronize()

    # Calculate the average time per run in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / 100.0

# ==============================================================================
# Main Testing Function
# ==============================================================================

def test_sdpa_performance():
    """
    Tests and compares the performance of SDPA with math and flash backends.
    """
    # --- 1. Test Configuration ---
    # Using parameters that are common for LLMs and highlight performance differences
    BATCH_SIZE = 1
    SEQ_LENGTH = 8192*8
    NUM_HEADS = 32
    HEAD_DIM = 128
    D_MODEL = NUM_HEADS * HEAD_DIM
    DTYPE = torch.bfloat16
    DEVICE = "cuda"

    print("--- SDPA Performance Test ---")
    print(f"Configuration: Batch={BATCH_SIZE}, SeqLen={SEQ_LENGTH}, Heads={NUM_HEADS}, Dim={HEAD_DIM}")
    print(f"DataType: {DTYPE}")
    print("----------------------------------\n")

    # --- 2. Model and Input Setup ---
    # Create a model configuration
    model_config = ModelConfig(
        d_model=D_MODEL,
        n_heads=NUM_HEADS,
        flash_attention=True # This enables the use of flash_attn_func if available
    )
    
    # Instantiate a transformer block. We only need it to access its sdpa method.
    # The block itself doesn't need to be fully initialized with weights for this test.
    try:
        transformer_block = LLaDASequentialBlock(layer_id=0, config=model_config, cache=BufferCache()).to(DEVICE)
        transformer_block.eval() # Set to eval mode to disable dropout
    except Exception as e:
        print(f"Error initializing LLaDASequentialBlock: {e}")
        print("Please ensure modeling_llada.py and configuration_llada.py are in the same directory.")
        return

    # Create random input tensors (Query, Key, Value)
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # --- 3. Benchmarking ---
    
    # Because sdp_kernel is not provided, we will use a simple context manager
    # to enable/disable flash attention via torch's backend context manager.
    # This is the standard way to control SDPA implementations.
    from torch.backends.cuda import sdp_kernel

    print("Benchmarking SDPA with 'math' backend (equivalent to `enable_flash=False`)...")
    with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        # We wrap the call in a lambda to pass it to the measurement function
        math_time = measure_time(
            lambda: transformer_block._scaled_dot_product_attention(q, k, v)
        )
    print(f"Average time for 'math' backend: {math_time:.4f} ms\n")
    math_time=0
    print("Benchmarking SDPA with 'flash' backend (equivalent to `enable_flash=True`)...")
    try:
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            flash_time = measure_time(
                lambda: transformer_block._scaled_dot_product_attention(q, k, v)
            )
        print(f"Average time for 'flash' backend: {flash_time:.4f} ms\n")
        
        # --- 4. Results ---
        print("--- Results ---")
        if flash_time > 0:
            speedup = math_time / flash_time
            print(f"Flash Attention is {speedup:.2f}x faster than the Math backend.")
        else:
            print("Could not calculate speedup due to zero execution time for flash.")
        print("---------------")

    except RuntimeError as e:
        if "Flash Attention" in str(e):
            print("Flash Attention is not available on this hardware or with this version of PyTorch.")
            print("Test for 'flash' backend skipped.")
        else:
            raise e


if __name__ == '__main__':
    test_sdpa_performance()