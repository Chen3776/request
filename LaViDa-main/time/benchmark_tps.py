import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HF_HUB_OFFLINE"] = '1'

import torch
from PIL import Image
import copy
from torch.profiler import profile, record_function, ProfilerActivity

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

import time
import numpy as np

def run_tps_benchmark():

    print("--- 1. Configuring benchmark parameters... ---")
    MAX_NEW_TOKENS = 5000
    WARMUP_RUNS = 1
    BENCHMARK_RUNS = 5
    use_prefix_lm = False
    
    generation_params = {
        'steps': MAX_NEW_TOKENS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'block_length': MAX_NEW_TOKENS//2,
        'temperature': 0.,
        'cfg_scale': 0.,
        'remasking': 'low_confidence'
    }
    print(f"Benchmark settings: {MAX_NEW_TOKENS=}, {WARMUP_RUNS=}, {BENCHMARK_RUNS=}")
    print("----------------------------------------------------")

    print("--- 2. Loading model and tokenizer... ---")

    pretrained = "/mnt/public/chenpengyi/models/lavida-L-ins"
    vision_tower_path = "/mnt/public/chenpengyi/models/siglip-so400m-patch14-384"
    model_name = "llava_llada" # 您的模型类型
    device = "cuda"
    device_map = "cuda"

    vision_kwargs = dict(
        mm_vision_tower=vision_tower_path,
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map,
        vision_kwargs=vision_kwargs, torch_dtype='bfloat16')

    model.eval()
    model.tie_weights()
    
    print("--- 3. Preparing input data... ---")
    conv_template = "llada"
    question = DEFAULT_IMAGE_TOKEN + "\nWrite a story based on the picture. The word count is not limited."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    image = Image.open('/mnt/public/chenpengyi/LaViDa-main/images/dante.png').convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    image_sizes = [image.size]
    
    prompt_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    print("--- Input data ready ---")
    
    decode_steps = MAX_NEW_TOKENS
    sparse_params_flex = {'whole_steps': MAX_NEW_TOKENS // 2,'skip': 0.2, 'select': 0.1, 'block_size': 128, 'new_generation': MAX_NEW_TOKENS}
    
    print(f"--- 4. Running {WARMUP_RUNS} warm-up iteration(s)... ---")
    for i in range(WARMUP_RUNS):
        _ = model.generate(
                inputs=prompt_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                max_new_tokens=decode_steps,
                block_length=decode_steps//2,
                do_sample=False,
                prefix_lm=use_prefix_lm,
                # SparseD_param=sparse_params_flex,
            )
    print("--- Warm-up complete ---")
    print("----------------------------------------------------")

    print(f"--- 5. Starting benchmark ({BENCHMARK_RUNS} iterations)... ---")
    # timings = []
    # generated_ids_for_display = None

    # for i in range(BENCHMARK_RUNS):
    #     torch.cuda.synchronize()
        
    #     start_time = time.perf_counter()
    #     generated_ids = model.generate(
    #             inputs=prompt_ids,
    #             images=image_tensor,
    #             image_sizes=image_sizes,
    #             max_new_tokens=decode_steps,
    #             block_length=decode_steps//2,
    #             do_sample=False,
    #             prefix_lm=True,
    #             SparseD_param=sparse_params_flex,
    #         )

    #     torch.cuda.synchronize()
    #     end_time = time.perf_counter()
        
    #     iteration_time = end_time - start_time
    #     timings.append(iteration_time)
    #     print(f"Iteration {i+1}/{BENCHMARK_RUNS}: {iteration_time:.4f} seconds")
        
    #     if i == BENCHMARK_RUNS - 1:
    #         generated_ids_for_display = generated_ids

    print("--- Benchmark complete ---")
    print("----------------------------------------------------")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(
                inputs=prompt_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                max_new_tokens=decode_steps,
                block_length=decode_steps//2,
                do_sample=False,
                prefix_lm=use_prefix_lm,
                # SparseD_param=sparse_params_flex,
            )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    iteration_time = end_time - start_time
    

    # =================================================================
    # 6. 计算并报告结果
    # =================================================================
    print("\n========== TPS Benchmark Results ==========")
    
    # avg_time = np.mean(timings)
    # std_time = np.std(timings)
    # tps = MAX_NEW_TOKENS / avg_time
    tps = MAX_NEW_TOKENS / iteration_time

    # print(f"Average generation time: {avg_time:.4f} seconds")
    # print(f"Standard deviation:      {std_time:.4f} seconds")
    print(f"Generation time: {iteration_time:.4f} seconds")
    print(f"Tokens per second (TPS): {tps:.2f} tokens/sec")
    print("==========================================")


    # =================================================================
    # 7. 解码并打印最后一次的生成结果以供验证
    # =================================================================
    # print("\n--- Last Generation Output (for verification) ---")
    
    # generated_text = tokenizer.batch_decode(
    #     generated_ids_for_display[:, generated_ids.shape[1]:], 
    #     skip_special_tokens=True
    # )[0]
    
    # print("\n========== Model Generation Result ==========")
    # print(generated_text)
    # print("===========================================")


if __name__ == '__main__':
    run_tps_benchmark()