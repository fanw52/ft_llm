'''
计算chatglm在微调数据中的最大字符长度,包括input和target
'''
import argparse

import jsonlines
import numpy as np

from models.baichuan.tokenization_baichuan import BaiChuanTokenizer
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="baichuan", type=str)
parser.add_argument('--data_path', default="/data/wufan/data/BELLE/Belle_open_source_0.5M.json", type=str)

args = parser.parse_args()

model_name = args.model_name

if model_name == "chatglm":
    model_name_or_path = "/data/wufan/llm/model/chatglm-6b"
    tokenizer = ChatGLMTokenizer.from_pretrained(model_name_or_path)

elif model_name == "baichuan":
    model_name_or_path = "/data/pretrained_models/baichuan_7b"
    tokenizer = BaiChuanTokenizer.from_pretrained(model_name_or_path)

else:
    raise Exception

input_length = []
target_length = []
total_length = []

max_input_length = 1280
max_target_length = 256
max_seq_length = max_input_length + max_target_length

special_data = []

c = 0

result = []
with jsonlines.open(args.data_path) as reader:
    for line in reader:
        instruction = line.get("instruction", "")
        input = line.get("input", "")
        target = line.get("output", "")

        a_ids = tokenizer.encode(text=instruction+input, add_special_tokens=False)
        b_ids = tokenizer.encode(text=target, add_special_tokens=False)


        input_length.append(len(a_ids))
        target_length.append(len(b_ids))
        total_length.append(len(a_ids)+len(b_ids))
        result.append(line)

sorted_input_length = sorted(input_length)
sorted_target_length = sorted(target_length)
sorted_total_length = sorted(total_length)

nums = len(sorted_total_length)

print(c)
percent_list = np.linspace(0.95, 0.99, 5)

for percent in percent_list:
    print(
        f"percent: {percent}, "
        f"输入最大长度: {sorted_input_length[int(percent * nums)]}, "
        f"目标最大长度: {sorted_target_length[int(percent * nums)]}, "
        f"总共最大长度: {sorted_total_length[int(percent * nums)]}")

# def save_jsonl(data,path):
#     with jsonlines.open(path,'w') as w:
#         for line  in data:
#             w.write(line)
#
# save_jsonl(result,output_path)
