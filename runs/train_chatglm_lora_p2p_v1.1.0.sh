# 在v1.0.0的基础上增加长尾3141数据, 共8703
lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=1e-5
model_name_or_path="/data/pretrained_models/chatglm6b"   # LLM底座模型路径，或者是huggingface hub上的

your_data_path="/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/chatglm_relation_p2p_fine_grit_v2_v0.6"  # 填入数据集所在的文件夹路径
your_checkpopint_path="/data2/wufan/llm/experiments/p2p_relation_v1.1.0"  # 填入用来存储模型的路径

# deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json
# --deepspeed ${deepspeed_config_file} \
# 激活a40环境chatglm
# conda activate chatglm
#torchrun \
#    --nnodes 1 \
#    --nproc_per_node 1 \
#    --master_port=29600 \

CUDA_VISIBLE_DEVICES=0 python \
    ft_chatglm_lora/main.py \
    --do_train \
    --train_file $your_data_path/train_add_tail.json \
    --validation_file $your_data_path/valid.json \
    --cache_dir $your_data_path/data_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/chatglm-6b-lora-wx-$LR \
    --overwrite_output_dir \
    --max_source_length 375 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_steps 3000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --seed 0 \
    --fp16 \

#    --overwrite_cache