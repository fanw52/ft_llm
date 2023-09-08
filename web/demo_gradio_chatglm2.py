import streamlit as st
from streamlit_chat import message
from transformers import AutoModel,AutoTokenizer
import torch

st.set_page_config(
    page_title="chatglm2",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    model_path = "/data/pretrained_models/chatglm2-6b-20230625"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_8bit=False,torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = model.eval()
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
            for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                                       temperature=temperature):
                query, response = history[-1]
                st.write(response)

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
    'top_p', 0.0, 1.0, 0.9, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.5, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
