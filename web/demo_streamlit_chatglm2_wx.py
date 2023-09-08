import json
import torch
import streamlit as st # 1.26.0
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
import sys


sys.path.append("/")
from models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
st.set_page_config(page_title="ChatGLM2-6B-Chat-WX")
st.title("ChatGLM2-6B-Chat-WX")


@st.cache_resource
def init_model():
    model_path = "/data/pretrained_models/chatglm2-6b-20230625"
    peft_model_path = "/data1/wufan/experiments/llm/wx_bilu_v2.1.0/chatglm-6b-lora-wx-1e-5/checkpoint-1000"

    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, peft_model_path, device_map='auto')

    tokenizer = ChatGLMTokenizer.from_pretrained(
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
9.提问5~10个问题，每个问题不少于15字

"""
    return f"""[Round 0]\n问:{instruction}###Input:\n{input}\n答："""

def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是ChatGLM2微调的讯问指引大模型，很高兴为您服务🥰")

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
        # messages.append({"role": "user", "content": prompt})
        # print(json.dumps({"role": "user", "content": prompt},ensure_ascii=False),flush=True)
        print(f"[user]:\n {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            instruction = generate_prompt(prompt)
            for response, history in model.stream_chat(tokenizer, instruction, history = []):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        # messages.append({"role": "assistant", "content": response})
        print(f"[assistant]:\n {response}\n\n", flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
