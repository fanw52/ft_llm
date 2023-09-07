FROM llm-server-base:v1.0.0 AS base
WORKDIR /app

COPY ./  /app

RUN pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN pip install streamlit streamlit_chat -i https://mirror.baidu.com/pypi/simple

RUN pip uninstall transformer-engine -y