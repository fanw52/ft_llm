import jsonlines
import torch
from peft import PeftModel

from models.baichuan.modeling_baichuan import BaiChuanForCausalLM as AutoModelForCausalLM
from models.baichuan.tokenization_baichuan import BaiChuanTokenizer as AutoTokenizer


#### 配置部分
base_model = "/data/pretrained_models/baichuan_7b"
lora_model = "/data/wufan/experiments/llm/baichuan/cls_jqtb_v1.0.0/baichuan-7b-cls-lora-1e-5/checkpoint-3000"
device = "cuda:0"
path = "/data/wufan/data/jqtb_cls/valid.json"
output_path = "/data/wufan/experiments/llm/baichuan/cls_jqtb_v1.0.0/valid.json"

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=1536
)

####


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# 注意：在训练初期，以8bit加载会导致预测结果都是unk
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,  # 不设置默认加载float32, 显存占用会很大
    low_cpu_mem_usage=True,
    device_map=device,  # 如果设置成auto可能会报错
    # trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model,
                                  lora_model,
                                  torch_dtype=torch.float16,
                                  device_map=device)

model.eval()

result = []
with torch.no_grad():
    with jsonlines.open(path) as reader:
        print("Start inference.")
        for line in reader:
            results = []
            input_text = line["input"]
            inputs = tokenizer(input_text, return_tensors="pt")

            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.unk_token_id,
                **generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s, skip_special_tokens=True)
            input_text_decode = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            response = output.replace(input_text_decode, "").strip()
            print(f"Input: {input_text_decode}\n")
            print(f"Output: {response}\n")
            result.append({"input": input_text, "target": line["target"], "answer": response})

with  jsonlines.open(output_path, 'w') as w:
    for line in result:
        w.write(line)
