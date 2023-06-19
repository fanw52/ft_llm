import torch
from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
from models.baichuan.modeling_baichuan import BaiChuanForCausalLM as AutoModelForCausalLM
from models.baichuan.tokenization_baichuan import BaiChuanTokenizer as AutoTokenizer

base_model = "/data/pretrained_models/baichuan_7b"
lora_model = "/data/wufan/experiments/llm/baichuan/baichuan_BELLE_v1.0.0/PromptCBLUE-chatglm-6b-lora-1e-5/checkpoint-6000"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

device = "cpu"
# 注意：在训练初期，以8bit加载会导致预测结果都是unk
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    # torch_dtype=torch.float16,  # 不设置默认加载float32, 显存占用会很大
    low_cpu_mem_usage=True,
    device_map=device,  # 如果设置成auto可能会报错
    # trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model,
                                  lora_model,
                                  # torch_dtype=torch.float16,
                                  device_map=device)

model.eval()

with torch.no_grad():
    print("Start inference.")
    results = []
    input_text = "我什么时候能暴富"
    inputs = tokenizer(input_text, return_tensors="pt")
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=1024
    )
    generation_output = model.generate(
        input_ids=inputs["input_ids"].to(device),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **generation_config
    )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)

    response = output
    print(f"Input: {input_text}\n")
    print(f"Output: {response}\n")
