import argparse

import torch
from peft import PeftModel
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--bits', default=4, type=int)
parser.add_argument('--model_name_or_path', default="/data1/pretrained_models/chatglm-6b-20230523", type=str)
parser.add_argument('--peft_name_or_path', default="/data1/wufan2/llm/experiments/wx_bilu_v2.0.3", type=str)
parser.add_argument('--quantize_dir', default="/data1/pretrained_models/chatglm-6b-20230523-int4", type=str)

args = parser.parse_args()

model_path = args.model_name_or_path
peft_model_path = args.peft_name_or_path
quantize_dir = args.quantize_dir


base_model = ChatGLMForConditionalGeneration.from_pretrained(model_path,device_map="auto")
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)

if peft_model_path:
    lora_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    base_model = lora_model.merge_and_unload()

base_model = base_model.quantize(args.bits)
base_model.save_pretrained(quantize_dir)
tokenizer.save_pretrained(quantize_dir)
