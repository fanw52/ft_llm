lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/data/wufan/llm/model/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="/data/wufan/data/wx_bilu_aug"  # 填入数据集所在的文件夹路径
your_checkpopint_path="/data/wufan/llm/experiments/wx_bilu_v2.0.2"  # 填入用来存储模型的路径

deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json

torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    ft_chatglm_lora/main.py \
    --do_train \
    --deepspeed ${deepspeed_config_file} \
    --train_file $your_data_path/train_aug_0530.json \
    --validation_file $your_data_path/val_aug_0530.json \
    --cache_dir $your_data_path/data_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/chatglm-6b-lora-wx-$LR \
    --overwrite_output_dir \
    --max_source_length 1280 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 20000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --seed 0