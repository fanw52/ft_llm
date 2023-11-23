lora_rank=8
lora_trainable="W_pack,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1

# 学习率太大可能会出现尖峰现象，导致模型发散
LR=2e-5

# 预训练模型路径
model_name_or_path="/data1/pretrained_models/Baichuan2-13B-Chat"

# 数据路径
your_data_path="/data/wufan/data/wx_bilu_plus"

# 保存路径
your_checkpopint_path="/data1/wufan/experiments/llm/Baichuan2-13B-Chat/wx_bilu_plus_20231122"

# deepspeed配置路径
deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json

# 启动指令
CUDA_VISIBLE_DEVICES=1 torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_port=29601 \
    ft_baichuan2_lora/run_sft_baichuan.py \
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
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16



