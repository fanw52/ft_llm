import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_model_path = "/data/pretrained_models/Baichuan2-13B-Base"
lora_model_path = "/data/wufan/experiments/llm/baichuan/pretrain_v1.0.0/baichuan2-lora-1e-5/checkpoint-10000"

generation_config = dict(
    temperature=1.,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_length=2048
)

#### 加载模型以及分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.user_token_id = 195
tokenizer.assistant_token_id = 196
device = "auto"

# # 注意：在训练初期，以8bit加载会导致预测结果都是unk
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,  # 不设置默认加载float32, 显存占用会很大
    low_cpu_mem_usage=True,
    device_map=device,  # 如果设置成auto可能会报错
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model,
                                  lora_model_path,
                                  torch_dtype=torch.float16,
                                  device_map=device)

model.eval()
## 预测
result = []
input_text = "类型#上衣*版型#宽松*风格#街头*风格#休闲*风格#朋克*图案#字母*图案#文字*图案#印花*衣样式#卫衣*衣款式#连帽*衣款式#"
with torch.no_grad():
    inputs = tokenizer.encode(input_text, add_special_tokens=False)
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
    print(response)
