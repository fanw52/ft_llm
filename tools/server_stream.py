import json
import os

import flask
from flask import Flask, request,stream_with_context

from models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm2.tokenization_chatglm import ChatGLMTokenizer

model_path = "/data/pretrained_models/chatglm2-6b-int4"
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()


app = Flask(__name__)

@app.route("/chatgpt", methods=["POST", "GET"])
def chatgpt():
    data = request.json
    question = data["question"]

    def stream():
        old_response = ""
        for response, history in model.stream_chat(tokenizer, question, ):
            query, response = history[-1]
            new_token = response.replace(old_response, "")
            old_response = response
            yield json.dumps({"token": new_token},ensure_ascii=False)

    return flask.Response(stream_with_context(stream()), mimetype="application/json")

app.run(host="0.0.0.0", port=5001, debug=True)


'''

import json

import requests


payload = {
    "messages": [{"role": "user", "content": "你好"}],
}
headers = {
    'Content-Type': 'application/json'
}

url = "http://192.168.51.17:5001/chat"

response = requests.request("POST", url, headers=headers, data=json.dumps(payload), stream=True)
# response = requests.request("POST", url, headers=headers, data=json.dumps(payload),)

if response.status_code == 200:
    for line in response:
        if line:
            try:
                json_data = json.loads(line)
                print(json_data)
            except Exception as e:
                print(e)
else:
    print(f"请求失败：{response.status_code}")

'''
