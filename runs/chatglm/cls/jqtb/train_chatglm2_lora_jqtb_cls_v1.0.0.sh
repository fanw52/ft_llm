lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-5

# a40###
model_name_or_path="/data/pretrained_models/chatglm2-6b-20230625"

your_data_path="/data/wufan/data/jqtb_cls"
######

# v100s
#model_name_or_path="/data/pretrained_models/chatglm-6b-20230523"
#your_data_path="/data/wufan/data/BELLE"

# 在当前配置下a40上减少40h训练时间,chatglm35G,178h,chatglm2占用27G，132h,训练时间上减少了25%左右，显存上减少23%

your_checkpopint_path="/data/wufan/experiments/llm/chatglm2/cls_jqtb"
deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json

CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=29611 \
    --nnodes 1 \
    --nproc_per_node 1 \
    ft_chatglm_lora/run_sft_chatglm.py \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --train_file $your_data_path/trainv2.json \
    --cache_dir $your_data_path \
    --chatglm2 True \
    --instruction_column instruction \
    --input_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/chatglm2-6b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 1280 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 1500 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16
