import torch

from models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
from transformers import AutoModel

model_path = "/data/pretrained_models/chatglm2-6b-20230625"
model_path = "/data/pretrained_models/llama2/"

torch.cuda.reset_max_memory_allocated()
# model = AutoModel.from_pretrained(model_path).quantize(8).half().cuda()
model = ChatGLMForConditionalGeneration.from_pretrained(model_path,device_map="auto")
mem = torch.cuda.max_memory_allocated()
mem = mem / 1024 / 1024 / 1024
print(f"显存消耗:{mem:.03f}G")

tokenizer = ChatGLMTokenizer.from_pretrained(model_path)

in_text = "what is your name"
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
