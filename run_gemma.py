# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/data/pretrained_models/gemma-7b-it", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/data/pretrained_models/gemma-7b-it", device_map="auto",
                                             trust_remote_code=True, torch_dtype=torch.float16)

input_texts = ["Write me a poem about Machine Learning.", "1+1", "1+1="]
for input_text in input_texts:
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids,max_length=2048)
    print(tokenizer.decode(outputs[0]))
