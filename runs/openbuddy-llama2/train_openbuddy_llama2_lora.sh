lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-5

model_name_or_path="/data1/pretrained_models/openbuddy-llama2-13b-v8.1-fp16"

your_data_path="/data/wufan/data/wx_bilu_plus"

your_checkpopint_path="/data1/wufan/experiments/llm/openbuddy-llama2-13b-v8.1-fp16/wx_bilu_plus"
deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json

#CUDA_VISIBLE_DEVICES=1
torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    ft_llama_lora/run_sft_llama.py \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --train_file $your_data_path/train.json \
    --cache_dir $your_data_path \
    --instruction_column instruction \
    --input_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path \
    --overwrite_output_dir \
    --max_source_length 1536 \
    --max_target_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16



