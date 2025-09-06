

export OPENAI_API_URL="http://123.129.219.111:3000/v1/chat/completions"
export OPENAI_API_KEY="sk-dRTI4t9lMHu6xrRhSEU0jaM284a1RflVS4RpqCQEsmAviFU7"
export YOUR_API_KEY="sk-dRTI4t9lMHu6xrRhSEU0jaM284a1RflVS4RpqCQEsmAviFU7"

export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export DEBUG_PRINT_IMAGE_RES=1

MODEL_PATH='/remote-home/pengyichen/Omni/DLLM/models/lavida-D-ins'

accelerate launch --num_processes=4 --main_process_port=11131 \
    -m lmms_eval \
    --model llava_dream \
    --model_args pretrained=$MODEL_PATH,conv_template=dream,model_name=llava_dream \
    --tasks mathvista_testmini \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True,max_new_tokens=2 \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ./logs/ --verbosity=DEBUG \