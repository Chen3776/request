import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import torch
import copy

os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["HF_HUB_OFFLINE"]='1'
os.environ["HF_DATASETS_CACHE"]="/mnt/public/chenpengyi/hf_datasets_cache"

# --- 配置信息 ---
pretrained = "/mnt/public/chenpengyi/models/lavida-L-ins"
model_name = "llava_llada"
device = "cuda"
device_map = "cuda"
conv_template = "llada"
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in detail."
DECODE_STEPS = 64

# --- 在模型中启用绘图功能 ---
# 这个环境变量会被你的模型代码检查
os.environ["PLOT_ATTENTION"] = "1"

# --- 模型和分词器加载 ---
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print("正在使用的Prompt:")
print(prompt_question)

vision_kwargs = dict(
    mm_vision_tower="/mnt/public/chenpengyi/models/siglip-so400m-patch14-384",
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
model.to(torch.bfloat16)

# --- 图像和输入准备 ---
image = Image.open('images/dante.png').convert('RGB')
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

# --- 多步解码循环 ---
print(f"\n开始进行 {DECODE_STEPS} 步解码并绘制注意力图...")

with torch.no_grad():
    for step in range(DECODE_STEPS):
        # 为当前解码步骤设置环境变量
        os.environ["DECODE_STEP"] = str(step)
        
        print(f"  - 正在为第 {step+1}/{DECODE_STEPS} 步生成token...")

        # 准备当前步骤的模型输入
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=image_tensor,
            image_sizes=image_sizes,
            return_dict=True
        )

        # 贪心解码：获取最后一个token的logits，并选择最可能的token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        # 将新生成的token拼接到input_ids中，用于下一次迭代
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        # 如果生成了句子结束符，则停止
        # import pdb
        # pdb.set_trace()
        if next_token[0].item() == tokenizer.eos_token_id:
            print("已生成句子结束符，停止解码。")
            continue

print("\n解码和绘图完成。")
# generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
# print("\n最终生成的文本:")
# print(generated_text)

# --- 清理工作 ---
# 使用后最好删除环境变量
del os.environ["PLOT_ATTENTION"]
if "DECODE_STEP" in os.environ:
    del os.environ["DECODE_STEP"]