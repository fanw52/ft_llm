import os
import sys
import time

import jsonlines
from peft import PeftModel
from transformers import set_seed

{
    "": "报警人称许兰心16岁女身高160偏瘦穿黑裙子蓝衬衣说去新华书店，去找了没在，现在都没回家"
}
sys.path.append("./")
from chatglm2.models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2.models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default="/data/wufan/data/jqtb_cls", type=str)
parser.add_argument('--out_path', default="/data/wufan/experiments/llm/chatglm/cls_jqtb", type=str)
parser.add_argument('--model_path', default="/data/wufan/llm/model/chatglm-6b", type=str)
parser.add_argument('--peft_path', default="", type=str)
args = parser.parse_args()
'''
CUDA_VISIBLE_DEVICES=0 python tools/cls/infer_cls_jqtb.py启动
'''
in_path = args.in_path
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

TP = 0
FP = 0
FN = 0
with jsonlines.open(in_path) as reader:
    for line in reader:
        in_text = line["instruction"] + line["input"]
        target = line["target"]
        batch = tokenizer(in_text, return_tensors="pt", max_length=max_length - 512, truncation=True)
        # target = tokenizer(target, return_tensors="pt")
        out = model.generate(
            input_ids=batch['input_ids'],
            max_length=max_length,
            do_sample=False
        )
        answer = tokenizer.decode(out[0][len(batch['input_ids'][0]):])
        print("预测:", answer)
        print("标签:", target)
        print()
        line["answer"] = answer
        line["target"] = target

        answer_list = [a for a in answer.split("\n") if a != ""]
        target_list = [a for a in target.split("\n") if a != ""]

        TP += len(set(answer_list).intersection(set(target_list)))
        FP += len(set(answer_list).difference(set(target_list)))
        FN += len(set(target_list).difference(set(answer_list)))

        result.append(line)
        if len(result) == -1:
            break

print(f"TP:{TP}\nFP:{FP}\nFN{FN}")

P = TP / (TP + FP+1e-8)
R = TP / (TP + FN+1e-8)
F1 = 2*P*R/(P+R+1e-8)
print(f"P:{P}\nR:{R}\nF1:{F1}")
print("总时间消耗", time.time() - st)

with jsonlines.open(outpath, 'w') as w:
    for line in result:
        w.write(line)
