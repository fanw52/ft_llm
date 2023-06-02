import torch

from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from transformers import AutoModel

model_path = "/data1/pretrained_models/chatglm-6b-20230523"

torch.cuda.reset_max_memory_allocated()
# model = AutoModel.from_pretrained(model_path).quantize(8).half().cuda()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
mem = torch.cuda.max_memory_allocated()
mem = mem / 1024 / 1024 / 1024
print(f"显存消耗:{mem:.03f}G")

tokenizer = ChatGLMTokenizer.from_pretrained(model_path)

in_text = "制定一份详细的五一出行计划"
torch.cuda.reset_max_memory_allocated()
batch = tokenizer(in_text, return_tensors="pt")
out = model.generate(
    input_ids=batch['input_ids'].cuda(),
    max_length=256,
    do_sample=False,
    top_p=0.7,
    temperature=0.96
)
mem = torch.cuda.max_memory_allocated()
mem = mem / 1024 / 1024 / 1024

print(f"显存消耗:{mem:.03f}G")

in_text_decode = tokenizer.decode(batch["input_ids"][0])

out_text = tokenizer.decode(out[0])
answer = out_text.replace(in_text_decode, "").replace("\nEND", "").strip()
print(answer)
