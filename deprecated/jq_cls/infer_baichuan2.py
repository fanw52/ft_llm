import jsonlines
import torch
from peft import PeftModel
import time
from transformers import  AutoModelForCausalLM,AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', default="/data/wufan/data/jqtb_cls", type=str)
parser.add_argument('--out_path', default="/data/wufan/experiments/llm/chatglm/cls_jqtb", type=str)
parser.add_argument('--model_path', default="/data/wufan/llm/model/chatglm-6b", type=str)
parser.add_argument('--peft_path', default="", type=str)
args = parser.parse_args()


#### 配置部分
base_model = args.model_path
lora_model = args.peft_path
device = "auto"
in_path = args.in_path
out_path = args.out_path

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.3,
    max_length=2048
)

####


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# 注意：在训练初期，以8bit加载会导致预测结果都是unk
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,  # 不设置默认加载float32, 显存占用会很大
    low_cpu_mem_usage=True,
    device_map=device,  # 如果设置成auto可能会报错
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model,
                                  lora_model,
                                  torch_dtype=torch.float16,
                                  device_map=device)

model.eval()



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
        batch = tokenizer(in_text, return_tensors="pt", max_length=max_length, truncation=True)
        out = model.generate(
            input_ids=batch['input_ids'].cuda(),
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

with jsonlines.open(out_path, 'w') as w:
    for line in result:
        w.write(line)

