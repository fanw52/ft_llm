lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.05
LR=5e-5
#model_name_or_path="/data/wufan/llm/model/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称

# a40###
#model_name_or_path="/data/pretrained_models/chatglm-6b-20230524"
#model_name_or_path="/data/pretrained_models/chatglm2-6b-20230625"

#your_data_path="/data/wufan/data/BELLE"
######

# v100s
model_name_or_path="/data/pretrained_models/chatglm2-6b-20230625"
your_data_path="/data/wufan/data/CCKS2023-PromptCBLUE中文医疗大模型评测基准—开源赛道/01_比赛数据集"

# 在当前配置下a40上减少40h训练时间,chatglm35G,178h,chatglm2占用27G，132h,训练时间上减少了25%左右，显存上减少23%

your_checkpopint_path="/data/wufan/experiments/train_chatglm2_lora_v1.0"
deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json
#     --cache_dir $your_data_path \
torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    ft_chatglm_lora/run_sft_chatglm.py \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --train_file $your_data_path/merge_train_and_dev.json \
    --chatglm2 True \
    --instruction_column instruction \
    --input_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 7500 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16



