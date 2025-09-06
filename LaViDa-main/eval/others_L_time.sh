
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="/mnt/public/chenpengyi/hf_datasets_cache"
export LLADA_VISION_ENCODER="/mnt/public/chenpengyi/models/siglip-so400m-patch14-384"

export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1

# gqa,realworldqa,mathvision_testmini,ai2d,scienceqa_img,mme,mmmu_val,flickr30k_test_lite,coco2017_cap_val_lite,textvqa_val_lite,docvqa_val_lite,chartqa_lite,infovqa_val_lite

MODEL_PATH='/mnt/public/chenpengyi/models/lavida-L-ins'

accelerate launch --num_processes=1 --main_process_port=11171 \
    -m lmms_eval \
    --model llava_llada \
    --model_args pretrained=$MODEL_PATH,conv_template=llada,model_name=llava_llada \
    --tasks coco2017_cap_val_lite \
    --batch_size 1 \
    --gen_kwargs alg=topk_margin,prefix_lm=True,max_new_tokens=32 \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ./logs/ --verbosity=DEBUG \