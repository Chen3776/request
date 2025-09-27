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

#  warmup?

def run_generate_with_profiler():

    ## 1. Model and Data Loading
    print("--- 1. Loading model and tokenizer... ---")
    pretrained = "/mnt/public/chenpengyi/models/lavida-L-ins"
    vision_tower_path = "/mnt/public/chenpengyi/models/siglip-so400m-patch14-384"
    model_name = "llava_llada"
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
        vision_kwargs=vision_kwargs, torch_dtype='bfloat16', attn_implementation="eager"
    )
    model.eval()
    model.tie_weights()
    print("--- Model loading complete ---")

    ## 2. Input Data Preparation
    print("\n--- 2. Preparing input data... ---")
    conv_template = "llada"
    question = DEFAULT_IMAGE_TOKEN + "\nWrite a story based on the picture. The word count is not limited."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    image = Image.open('images/dante.png').convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    image_sizes = [image.size]

    prompt_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    print("--- Input data ready ---")

    ## 3. Execute Generation with Profiler
    decode_steps = 128
    print(f"\n--- 3. Starting model.generate for {decode_steps} steps with Profiler... ---")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("model.generate_call"):
            generated_ids = model.generate(
                inputs=prompt_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                max_new_tokens=decode_steps,
                do_sample=False,
                use_cache=False,
                prefix_lm=False,
            )

    print("--- Generation complete ---")
    
    key_averages_list = prof.key_averages()
    
    ## 4. Performance Analysis Results
    print("\n========== Full Performance Analysis (Top 20) ==========")
    print(key_averages_list.table(sort_by="device_time_total", row_limit=20))
    prof.export_chrome_trace("generation_trace.json")
    print("\nDetailed performance timeline exported to 'generation_trace.json'")
    print("==========================================================")

    ## 5. Custom Analysis for Attention + MLP Layers
    print("\n========== Custom Attention + MLP Timing Analysis ==========")
    
    # Correctly iterate through the list of FunctionEventAvg objects
    event_times_us = {evt.key: evt.device_time_total for evt in key_averages_list}
    event_counts = {evt.key: evt.count for evt in key_averages_list}
    
    
    # Get total GPU time from the profiler summary (in microseconds)
    # This value represents the sum of all "self" CUDA times.

    # Extract times for key components (convert from microseconds to milliseconds)
    # Use .get(key, 0) to handle cases where an event might not be present
    attention_core_time_ms = event_times_us.get('aten::scaled_dot_product_attention', 0) / 1000.0
    linear_layers_time_ms = event_times_us.get('aten::linear', 0) / 1000.0
    
    # Calculate the combined time as per your definition
    attn_and_mlp_total_time_ms = attention_core_time_ms + linear_layers_time_ms
    
    print(f"Core Attention calculation time (scaled_dot_product_attention): {attention_core_time_ms:.2f} ms")
    print(f"All Linear Layers time (aten::linear, for MHA & MLP): {linear_layers_time_ms:.2f} ms")
    print("---------------------------------------------------------------")
    print(f"Combined 'Attention + MLP' Total Time: {attn_and_mlp_total_time_ms:.2f} ms")
    print("===========================================================")
    
    
    reuse_attention_time_ms = event_times_us.get("reuse_attention):", 0) / 1000.0
    print(f"reuse_attention Time: {reuse_attention_time_ms:.2f} ms, event count: {event_counts.get('reuse_attention', 0)}")
    print("===========================================================")
    
    ## 6. Decode and Print Final Text Result
    print("\n--- 6. Decoding and printing final result ---")
    
    generated_text = tokenizer.batch_decode(
        generated_ids[:, prompt_ids.shape[1]:], 
        skip_special_tokens=True
    )[0]

    print("\n========== Model Generation Result ==========")
    print(generated_text)
    print("===========================================")

if __name__ == '__main__':
    run_generate_with_profiler()