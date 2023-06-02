import jsonlines
from peft import PeftModel
from tqdm import tqdm

from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

model_path = "/data/wufan/llm/model/chatglm-6b"
peft_path = "/data/wufan/llm/experiments/wx_bilu_v2.0.3/chatglm-6b-lora-wx-1e-5"

model = ChatGLMForConditionalGeneration.from_pretrained(model_path)
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
# model = PeftModel.from_pretrained(model, peft_path)
model.half().cuda()

path = "/data/wufan/data/wx_bilu_aug/val_aug_0531.json"
out_path = "/data/wufan/data/wx_bilu_aug/val_aug_0531_pred_raw.json"

result = []
c = 0
with jsonlines.open(path) as reader:
    for line in tqdm(reader):
        in_text = line["input"]
        batch = tokenizer(in_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch['input_ids'].cuda(),
            max_length=1536,
            do_sample=False,
            top_p=0.7,
            temperature=0.96
        )
        in_text_decode = tokenizer.decode(batch["input_ids"][0])

        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(in_text_decode, "").replace("\nEND", "").strip()
        print(answer)
        line["answer"] = answer
        result.append(line)


with jsonlines.open(out_path, 'w') as w:
    for line in result:
        w.write(line)
