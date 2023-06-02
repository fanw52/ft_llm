lora_rank=8
lora_trainable="c_attn"
modules_to_save="null" # 保存除了lora层之外可训练的模块
lora_dropout=0.1
LR=2e-4
model_name_or_path="gpt2-xl"
your_data_path="/data/wufan/data/AdvertiseGen"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/toy_examples_v1.0.0/"  # 填入用来存储模型的路径
cache_dir=/data/wufan/llm/cache
deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json
# #    --cache_dir ${cache_dir} \

#torchrun \
#    --nnodes 1 \
#    --nproc_per_node 2 \

CUDA_VISIBLE_DEVICES=0 python \
    ft_clm/run_pt_clm.py \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/gpt2-lora-$LR \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 300 \
    --block_size 512 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16
