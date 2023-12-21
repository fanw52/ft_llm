import jsonlines
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#### 配置部分
base_model = "/data/pretrained_models/Baichuan2-13B-Chat"
lora_model = "/data1/wufan/experiments/llm/Baichuan2-13B-Chat/wx_bilu_plus_20231122/checkpoint-3000"
path = "/data/wufan/data/wx_bilu_plus/train.json"
# output_path = "/data1/wufan/experiments/llm/Baichuan2-13B-Chat/wx_bilu_plus_20231122/wx/train_prediction.json"


# 会出现较多比较重复性的提问，比如：
'''
generation_config = dict(
    temperature=0.75,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    # repetition_penalty=1.3,
    max_new_tokens=1024
)
Output:  1. 你为什么报案？
2. 请描述一下具体事情发生的经过。
3. 张雨帆在赌局里做什么工作？
4. 你在赌局内向张雨帆借过钱吗？
5. 你和张雨帆之间有没有其他交易？
6. 你和张雨帆之间有没有其他纠纷？
7. 你和张雨帆之间有没有其他关系？
8. 你和张雨帆之间有没有其他合作？
9. 你和张雨帆之间有没有其他约定？
10. 你和张雨帆之间有没有其他争议？</s>
'''

generation_config = dict(
    temperature=0.75,
    top_k=40,
    top_p=0.9,
    do_sample=True, #
    num_beams=1,
    # repetition_penalty=1.3,  # 增加这一项后，预测的问题的数目会降低
    max_length=2048
)

#### 加载模型以及分词器
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.user_token_id = 195
tokenizer.assistant_token_id = 196
device = "auto"
# # 注意：在训练初期，以8bit加载会导致预测结果都是unk
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
## 预测
result = []
with torch.no_grad():
    with jsonlines.open(path) as reader:
        print("Start inference.")
        for line in reader:
            results = []
            instruction = line["instruction"]
            input_text = line["input"]
            input_text = instruction + input_text
            inputs = tokenizer.encode(input_text,add_special_tokens=False)
            inputs = [tokenizer.user_token_id] + inputs + [tokenizer.assistant_token_id]
            inputs = torch.LongTensor([inputs])
            generation_output = model.generate(
                input_ids=inputs.to("cuda"),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_config
            )
            s = generation_output[0]

            response = tokenizer.decode(s[len(inputs[0]):], skip_special_tokens=False)

            print(f"Input: {input_text}\n")
            print(f"Output: {response}\n")
            result.append({"input": input_text, "target": line["target"], "answer": response})
#
# with jsonlines.open(output_path, 'w') as w:
#     for line in result:
#         w.write(line)
