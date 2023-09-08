import sys

import streamlit as st
import torch
from streamlit_chat import message
from transformers import AutoTokenizer, GenerationConfig
from transformers import LlamaForCausalLM

st.set_page_config(
    page_title="代码生成模型",
    page_icon=":robot:"
)


def generate_prompt(instruction, input=None):
    return f"""<s>Human: {instruction}\n</s><s>Assistant: """


@st.cache_resource
def get_model():
    model_path = "/data1/pretrained_models/Llama2-Chinese-13b-Chat"
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_8bit=False,
                                                  torch_dtype=torch.float16,
                                                  device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return tokenizer, model


def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            prompts = generate_prompt(input, "")

            batch = tokenizer(prompts, return_tensors="pt")
            out = model.generate(
                input_ids=batch['input_ids'].cuda(),
                max_length=max_length,
                do_sample=False,
                top_p=top_p,
                temperature=temperature
            )
            in_text_decode = tokenizer.decode(batch["input_ids"][0])

            out_text = tokenizer.decode(out[0])
            answer = out_text.replace(in_text_decode, "").replace("<|endoftext|>", "").strip()

            st.write(answer)
            history.append((input, answer))
    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
                           height=100,
                           placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 2048, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
