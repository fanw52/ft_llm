import argparse
import json
import time

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default="/data/pretrained_models/chatglm2-6b-20230625", type=str)
parser.add_argument('--chatglm2', default=True, type=bool)
parser.add_argument('--batch', default=1, type=int)
parser.add_argument('--load_in_8bit', default=False, type=bool)

args = parser.parse_args()

model_path = args.model_name_or_path

if args.chatglm2:
    from models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
    from models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
else:
    from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
    from models.chatglm.tokenization_chatglm import ChatGLMTokenizer

# model = ChatGLMForConditionalGeneration.from_pretrained(model_path, device_map="cuda:0").half().cuda()
# model = model.eval()
# tokenizer = ChatGLMTokenizer.from_pretrained(model_path,)

tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
if args.load_in_8bit:
    model = ChatGLMForConditionalGeneration.from_pretrained(model_path).quantize(8).cuda()
else:
    model = ChatGLMForConditionalGeneration.from_pretrained(model_path).half().cuda()

model = model.eval()

def build_prompt(text):
    return "[Round {}]\n\n问：{}\n\n答：".format(1, text)


nums_tokens = 0

t1 = time.time()
with torch.no_grad():
    dataset = []
    entry = "/data/wufan/data/BELLE/Belle_open_source_0.5M.json"
    with open(entry, encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))

    correct = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch)
    for i, batch in enumerate(tqdm(dataloader)):
        queries = []
        instruction_batch = batch["instruction"]
        input_batch = batch["input"]
        output_batch = batch["output"]

        for instruction, input, output in zip(instruction_batch, input_batch, output_batch):
            query = build_prompt(instruction + input)
            queries.append(query)

        inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=2048)

        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            nums_tokens += len(output)
            # response = tokenizer.decode(output)
            # intermediate_outputs.append(response)

        if i == 20:
            break

t = time.time() - t1
print(nums_tokens, t, nums_tokens / t)
