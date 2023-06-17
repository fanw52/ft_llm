
lora_rank=8
lora_trainable="W_pack,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/data/pretrained_models/baichuan_7b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="/data/wufan/llm/PromptCBLUE/datasets/PromptCBLUE/toy_examples"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/toy_examples_v1.0.0/"  # 填入用来存储模型的路径

deepspeed_config_file=./configs/deepspeed_config_zero2_offload.json

torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    ft_baichuan_lora/run_sft_baichuan.py \
    --deepspeed ${deepspeed_config_file} \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 828 \
    --max_target_length 196 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 300 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout}



