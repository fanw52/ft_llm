import json

import streamlit as st  # 1.26.0
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

st.set_page_config(page_title="OpenBuddy-13B-Chat-WX")
st.title("OpenBuddy-13B-Chat-WX")


@st.cache_resource
def init_model():
    model_path = "/data/pretrained_models/openbuddy-llama2-13b-v8.1-fp16"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_prompt(input):
    instruction = """###Instruction:

根据句子内容，针对句子中未提问的问题或者已经提到的事情进一步提问，返回几个的提问的结果，并满足如下几点要求：
1.如果在提问并回答如下话题：个人情况，个人简历，家庭成员，法律条款，身体状况等，返回的问题可以参考但不局限于：因为什么事情报案？描述一下具体事情发生的经过？
2.如果在提问并回答案件经过，需要依据人物，时间，地点，事件内容，补充句子中未提及的问题；
3.如果事发经过中，未提及事情发生的时间、地点，请补充提问；
4.不能提问与句子无关的内容；
5.不需要回答句子中的问题；
6.问题在对话中不能有答案；
7.问题需要对警察梳理案件有正向促进作用；
8.不能提问句子中已经存在或者相似的问题；
9.提问5~10个问题，每个问题不少于15字；
10.如果输入的内容无法理解，请回答：无法理解输入的内容，请重新组织语言

"""
    return f"""\nUser:{instruction}###Input:\n{input}\n\nAssistant:"""

def sample_top_p(probs, p):
    # probs: [bs, 1, vocab_size]

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


def stream_generate(model, tokenizer, instruction, max_length, top_p, temperature):
    """ 流式生成文本，页面交互体验更友好一些 """

    with torch.no_grad():
        batch = tokenizer(instruction, return_tensors="pt")
        in_tokens = batch['input_ids'].cuda()
        prompt_len = len(in_tokens[0])

        if len(in_tokens[0]) > max_length - 128:
            return f"输入token总长度超过了 max_len-128，输入token长度：{len(in_tokens[0])}，max_length：{max_length}。请清空历史数据开始新的对话。", ""

        end_token = torch.tensor([tokenizer.eos_token_id]).to(in_tokens.device)
        past_key_values = None
        out_tokens = None

        # 生成的文本达到最大长度时停止推理、遇到终止字符时停止推理
        pre_text = ""
        while (out_tokens is None) or (
                (out_tokens[0][-1] != end_token) and (prompt_len + out_tokens.size()[1] < max_length)):
            forward_result = model(input_ids=in_tokens, past_key_values=past_key_values, use_cache=True)
            logits = forward_result.logits
            past_key_values = forward_result.past_key_values

            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                out_token = sample_top_p(probs, top_p)
            else:
                out_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            in_tokens = out_token

            if out_tokens is None:
                out_tokens = out_token
            else:
                out_tokens = torch.cat([out_tokens, out_token], dim=-1)

            total_text = tokenizer.decode(out_tokens[0])

            new_token = total_text[len(pre_text):]
            pre_text = total_text
            yield new_token


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是OpenBuddy讯问指引大模型(指令版)，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()
    # TODO: 需要python3.8
    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user]:\n {prompt}", flush=True)
        #TODO: 每次不会传入历史对话信息
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            instruction = generate_prompt(prompt)
            pre_response = ""
            for response in stream_generate(model,tokenizer,instruction,max_length=2048,top_p=0.9, temperature=1):
                pre_response += response
                if "</s>" in pre_response:
                    pre_response = pre_response[:-4]
                placeholder.markdown(pre_response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        pre_response.replace("</s>", "")
        print(pre_response)
        messages.append({"role": "assistant", "content": pre_response})
        # print(json.dumps(messages, ensure_ascii=False), flush=True)
        print(f"[assistant]:\n {response}\n\n", flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
