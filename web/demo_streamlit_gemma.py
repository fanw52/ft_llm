import streamlit as st
import torch
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Gemma演示",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    model_path = "/data/pretrained_models/gemma-7b-it"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
    #                                              torch_dtype=torch.float16)
    # model = model.eval()
    # return tokenizer, model
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=dtype,
        trust_remote_code=True
    )
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
            # for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
            #                                    temperature=temperature):
            #     query, response = history[-1]
            #     st.write(response)
            #
            # outputs = model.generate(input_ids=input_ids['input_ids'].cuda(), max_length=2048)
            # # for response in model.chat(tokenizer, messages, stream=True):
            # response = tokenizer.decode(outputs[0][len(input_ids["input_ids"][0]):])
            chat = [
                {"role": "user", "content": input},
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=2048)
            response = tokenizer.decode(outputs[0])
            print(response)

            template = f"<bos><start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model"
            # template  = f'''<bos><start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model'''
            response = response.replace(template, "").replace("<eos>","")
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
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
