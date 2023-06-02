'''
计算chatglm在微调数据中的最大字符长度,包括input和target
'''
import jsonlines
import numpy as np

from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

model_name_or_path = "/data/wufan/llm/model/chatglm-6b"
tokenizer = ChatGLMTokenizer.from_pretrained(model_name_or_path)

input_path = "/data/wufan/data/wx_bilu_aug/val_aug_0530.json"
output_path =  "/data/wufan/data/wx_bilu_aug/val_aug_0531.json"


input_length = []
target_length = []
total_length = []

max_input_length = 1280
max_target_length = 256
max_seq_length = max_input_length + max_target_length

special_data = []

c =0

result = []
with jsonlines.open(input_path) as reader:
    for line in reader:
        input = line["input"]
        target = line["target"]
        a_ids = tokenizer.encode(text=input, add_special_tokens=False)
        b_ids = tokenizer.encode(text=target, add_special_tokens=False)
        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1

        pad_len = max_seq_length - len(input_ids)

        ######## 按照最大长度过滤
        if pad_len < 0:
            special_data.append(line)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            continue
        if len(a_ids)>=max_input_length or len(b_ids)>=max_target_length-1:
            c+=1
            continue
        ##### 按照最大长度过滤

        input_length.append(len(a_ids))
        target_length.append(len(b_ids))
        total_length.append(len(input_ids))
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