import jsonlines
from peft import PeftModel
from tqdm import tqdm
from transformers import set_seed
import time
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

# llm
# model_path = "/data/wufan/llm/model/chatglm-6b"
# peft_path = "/data/wufan/llm/experiments/wx_bilu_v2.0.3/chatglm-6b-lora-wx-1e-5"

model_path = "/data1/pretrained_models/chatglm-6b-20230523"
peft_path = "/data1/wufan2/llm/experiments/wx_bilu_v2.0.3"

set_seed(0)
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path,device_map="auto")
model = PeftModel.from_pretrained(model, peft_path,device_map="auto")
model.half()
model.eval()

path = "/data1/wufan/data/wx_bilu_aug/val_aug_0531.json"
out_path = "/data1/wufan/data/wx_bilu_aug/val_aug_0531_top200.json"

result = []
c = 0

st  = time.time()
with jsonlines.open(path) as reader:
    for line in tqdm(reader):
        in_text = line["input"]
        batch = tokenizer(in_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch['input_ids'],
            max_length=1536,
            do_sample=False,
            top_p=0.7,
            temperature=0.95
        )
        in_text_decode = tokenizer.decode(batch["input_ids"][0])

        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(in_text_decode, "").replace("\nEND", "").strip()
        print(answer)
        line["answer"] = answer
        result.append(line)
        if len(result)==200:
            break

print("总时间消耗",time.time()-st)

with jsonlines.open(out_path, 'w') as w:
    for line in result:
        w.write(line)
