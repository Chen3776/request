import os
import torch
from PIL import Image
import copy
import numpy as np

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates

# --- 配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HF_HUB_OFFLINE"] = '1'

# 定义模型、路径等参数
pretrained = "/mnt/public/chenpengyi/models/lavida-L-ins"
vision_tower_path = "/mnt/public/chenpengyi/models/siglip-so400m-patch14-384"
model_name = "llava_llada"
device = "cuda"
device_map = "cuda:0"

# --- 加载模型和 Tokenizer ---
vision_kwargs = dict(
    mm_vision_tower=vision_tower_path,
    mm_resampler_type=None,
    mm_projector_type='mlp2x_gelu',
    mm_hidden_size=1152,
    use_mm_proj=True
)
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map,
    vision_kwargs=vision_kwargs, torch_dtype='bfloat16', attn_implementation="eager"
)

model.eval()
model.tie_weights()

# --- 准备输入（Prompt和图像） ---
conv_template = "llada"
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

image = Image.open('images/dante.png').convert('RGB')
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
image_sizes = [image.size]

# 对初始的prompt进行token化
prompt_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# --- 迭代生成（双向）的主循环 ---
decode_steps = 64
mask_token_id = 126336 # 根据你的模型代码，这是[MASK]的ID

# 创建初始序列，包含 prompt 和 64 个 MASK token
# 形状为: (1, len(prompt) + 64)
masked_sequence = torch.full((1, decode_steps), mask_token_id, dtype=torch.long, device=device)
input_ids = torch.cat([prompt_ids, masked_sequence], dim=1)

# 开始循环生成/填充
for step in range(decode_steps):
    
    print(f"--- 正在处理步骤 {step + 1}/{decode_steps} ---")
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids), # 注意力需要覆盖整个序列
            labels=None,
            images=image_tensor,
            image_sizes=image_sizes,
            return_dict=True,
            use_cache=False, # <-- 明确关闭KV缓存
        )

    # 获取当前需要填充的 MASK 位置的 logits
    # MASK 的位置索引 = prompt的长度 + 当前步数
    prompt_len = prompt_ids.shape[1]
    logits_for_mask = outputs.logits[:, prompt_len + step, :] # 注意这里的索引调整
    predicted_token_id = torch.argmax(logits_for_mask, dim=-1)

    # 更新输入序列，为下一次迭代做准备
    # 用预测出的token替换掉当前步骤的 MASK token
    input_ids[0, prompt_len + step] = predicted_token_id.item()

print(f"\n处理完成。所有注意力图已保存在 '{output_base_dir}' 文件夹中。")

# --- 解码并打印最终生成的文本 ---
final_ids = input_ids[:, prompt_len:]
generated_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
print("\n最终生成的文本内容:")
print(generated_text)