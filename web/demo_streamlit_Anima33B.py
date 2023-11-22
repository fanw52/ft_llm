import streamlit as st
import torch
from streamlit_chat import message
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

st.set_page_config(
    page_title="Anima",
    page_icon=":robot:"
)


def generate_prompt(instruction):
    return instruction


@st.cache_resource
def get_model():
    base_model = "/data1/pretrained_models/Anima33B-merged"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
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
            prompts = generate_prompt(input)
            print(prompts)
            batch = tokenizer(prompts, return_tensors="pt").to("cuda")
            out = model.generate(**batch, max_new_tokens=512)
            in_text_decode = tokenizer.decode(batch["input_ids"][0])
            out_text = tokenizer.decode(out[0])
            answer = out_text.replace(in_text_decode, "").replace("<|endoftext|>", "").strip()
            print(answer)
            print()
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
