import os
import sys
import time

import jsonlines
from peft import PeftModel
from transformers import set_seed

sys.path.append("./")
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default="/data/wufan/data/jqtb_cls", type=str)
parser.add_argument('--out_path', default="/data/wufan/experiments/llm/chatglm/cls_jqtb", type=str)
parser.add_argument('--model_path', default="/data/wufan/llm/model/chatglm-6b", type=str)
parser.add_argument('--peft_path', default="", type=int)
args = parser.parse_args()
'''
CUDA_VISIBLE_DEVICES=0 python tools/cls/infer_cls_jqtb.py启动
'''
in_path = args.in_path

experiment_name = args.experiment_name
out_dir = f"{args.out_dir}/{experiment_name}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_path = args.model_path
peft_path = args.peft_path

set_seed(0)
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path, device_map="auto")
model = PeftModel.from_pretrained(model, peft_path, device_map="auto")
model.half().eval()

outpath = args.out_path

result = []

st = time.time()
max_length = 2048
with jsonlines.open(in_path) as reader:
    for line in reader:
        in_text = line["input"]
        target = line["target"]
        batch = tokenizer(in_text, return_tensors="pt", max_length=max_length-512, truncation=True)
        target = tokenizer(target, return_tensors="pt")
        out = model.generate(
            input_ids=batch['input_ids'],
            max_length=max_length,
            do_sample=False
        )
        in_text_decode = tokenizer.decode(batch["input_ids"][0])
        target_decode = tokenizer.decode(target["input_ids"][0])

        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(in_text_decode, "").replace("\nEND", "").strip()
        print(in_text_decode)
        print("预测:", answer)
        print("标签:", target_decode)
        print()
        line["answer"] = answer
        line["target"] = target_decode
        result.append(line)
        if len(result) == -1:
            break

print("总时间消耗", time.time() - st)

with jsonlines.open(outpath, 'w') as w:
    for line in result:
        w.write(line)
