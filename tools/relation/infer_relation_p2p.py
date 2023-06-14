import time

import jsonlines
from peft import PeftModel
from transformers import set_seed

from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

in_dir = "/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/chatglm_relation_p2p_fine_grit_v2_v0.6"

experiment_name = "p2p_relation_v1.1.0"
out_dir = f"/data/wufan/experiments/llm/chatglm/relation_p2p/{experiment_name}"

model_path = "/data/pretrained_models/chatglm6b"
peft_path = f"/data2/wufan/llm/experiments/{experiment_name}/chatglm-6b-lora-wx-1e-5"

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
        print(answer)
        print(target_decode)
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
