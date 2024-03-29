import time
import os
import jsonlines
from peft import PeftModel
from transformers import set_seed

from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default="/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/chatglm_relation_p2p_fine_grit_v2_v0.6", type=str)
parser.add_argument('--experiment_name', default="p2p_relation_v1.1.0", type=str)
parser.add_argument('--out_dir', default="/data/wufan/experiments/llm/chatglm/relation_p2p", type=str)
parser.add_argument('--model_path', default="/data/pretrained_models/chatglm6b", type=str)
parser.add_argument('--checkpoint_step', default=3000, type=int)
args = parser.parse_args()

in_dir = args.in_dir

experiment_name = args.experiment_name
out_dir = f"{args.out_dir}/{experiment_name}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_path = args.model_path
peft_path = f"/data2/wufan/llm/experiments/{experiment_name}/chatglm-6b-lora-wx-1e-5/checkpoint-{args.checkpoint_step}"

set_seed(0)
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path, device_map="auto")
model = PeftModel.from_pretrained(model, peft_path, device_map="auto")
model.half().eval()

path = f"{in_dir}/valid.json"
outpath = f"{out_dir}/valid.json"

result = []

st = time.time()
with jsonlines.open(path) as reader:
    for line in reader:
        in_text = line["input"]
        target = line["target"]
        batch = tokenizer(in_text, return_tensors="pt")

        target = tokenizer(target, return_tensors="pt")
        out = model.generate(
            input_ids=batch['input_ids'],
            max_length=512,
            do_sample=False,
            top_p=0.7,
            temperature=0.95
        )
        in_text_decode = tokenizer.decode(batch["input_ids"][0])
        target_decode = tokenizer.decode(target["input_ids"][0])

        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(in_text_decode, "").replace("\nEND", "").strip()
        print(in_text_decode)
        print("预测:",answer)
        print("标签:",target_decode)
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
