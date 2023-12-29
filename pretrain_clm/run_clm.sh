lora_rank=8
lora_trainable="W_pack,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1

# 学习率
LR=1e-5

# LLM底座模型路径，或者是huggingface hub上的模型名称
model_name_or_path="/data/pretrained_models/Baichuan2-13B-Base"

# 填入用来存储模型的路径
your_checkpopint_path="/data/wufan/experiments/llm/baichuan/pretrain_v1.0.0/"

# deepspeed配置
deepspeed_config_file="./config/ds_zero2_no_offload.json"

# 数据路径
# 预训练文本数据：字段text
your_data_path="/data/wufan/data/AdvertiseGen"

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    pretrain_clm/run_pt_clm.py \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --train_file $your_data_path/pretrain.json \
    --validation_file $your_data_path/pretrain.json \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/baichuan2-lora-$LR \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --block_size 2048 \
    --learning_rate $LR \
    --save_total_limit 1 \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16