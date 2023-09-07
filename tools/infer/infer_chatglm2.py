import torch
import json
from models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
import time
model_path = "/data1/pretrained_models/chatglm2-6b-20230625"

model = ChatGLMForConditionalGeneration.from_pretrained(model_path).half().cuda()

tokenizer = ChatGLMTokenizer.from_pretrained(model_path)

path = "/data/wufan/data/BELLE/Belle_open_source_0.5M_mini.json"
with open(path, encoding="utf-8") as rfile:
    data = json.load(rfile)
    num_prompts = len(data)
    t1 = time.time()
    REQUEST_LATENCY = []
    for line in data:
        print(line)
        in_text = line["conversations"][0]["value"]
        batch = tokenizer(in_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch['input_ids'].cuda(),
            max_length=256,
            do_sample=False,
            top_p=0.7,
            temperature=0.96
        )

        in_text_decode = tokenizer.decode(batch["input_ids"][0])

        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(in_text_decode, "").replace("\nEND", "").strip()
        REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
        print(answer)
    t2 = time.time()
    benchmark_time = t2 - t1
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    # avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    # print(f"Average latency: {avg_latency:.2f} s")
    # avg_per_token_latency = np.mean([
    #     latency / (prompt_len + output_len)
    #     for prompt_len, output_len, latency in REQUEST_LATENCY
    # ])
    # print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    # avg_per_output_token_latency = np.mean([
    #     latency / output_len
    #     for _, output_len, latency in REQUEST_LATENCY
    # ])
    # print("Average latency per output token: "
    #       f"{avg_per_output_token_latency:.2f} s")
