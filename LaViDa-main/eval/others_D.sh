
export OPENAI_API_KEY="sk-AnYR2xiz0LDKTEer8VDR19xHU4sNKoqqFwVFRU9zCCzQW5vu"
export OPENAI_API_URL="https://xiaoai.plus/v1"
export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"

export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1

MODEL_PATH='/remote-home/pengyichen/Omni/DLLM/models/lavida-D-ins'

accelerate launch --num_processes=6 --main_process_port=11121 \
    -m lmms_eval \
    --model llava_dream \
    --model_args pretrained=$MODEL_PATH,conv_template=dream,model_name=llava_dream \
    --tasks gqa,realworldqa,mathvision_testmini,ai2d,scienceqa_img,mme,mmmu_val,flickr30k_test_lite,coco2017_cap_val_lite,textvqa_val,docvqa_val_lite,chartqa_lite,infovqa_val_lite \
    --batch_size 1 \
    --gen_kwargs alg=topk_margin,prefix_lm=True \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ./logs/ --verbosity=DEBUG \