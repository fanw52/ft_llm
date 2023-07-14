# 警情分类数据,训练数据5951,验证数据1488

lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=1e-5
model_name_or_path="/data/wufan/llm/model/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="/data/wufan/data/jqtb_cls"

your_checkpopint_path="/data/wufan/llm/experiments/cls_jqtb_v1.0.0"  # 填入用来存储模型的路径

deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json

# 激活a40环境chatglm
# conda activate chatglm
torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_port=29601 \
    ft_chatglm_lora/run_sft_chatglm.py \
    --do_train \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --chatglm2 False \
    --train_file $your_data_path/trainv2.json \
    --cache_dir $your_data_path/data_cache \
    --overwrite_cache \
    --instruction_column instruction \
    --input_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/chatglm-6b-lora-wx-$LR \
    --overwrite_output_dir \
    --max_source_length 1280 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_steps 3000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --save_steps 10000 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --seed 0 \
    --fp16