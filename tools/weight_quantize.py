# from transformers import AutoModel
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
# model_path = "/data1/pretrained_models/chatglm-6b-20230523"
model_path = "/data1/pretrained_models/xxxx"
pt_save_directory = "/data1/pretrained_models/chatglm-6b-int4-test"


# model = AutoModel.from_pretrained(model_path,device_map='auto',trust_remote_code=True).quantize(4)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path,device_map='auto').quantize(4)
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model.save_pretrained(pt_save_directory)
tokenizer.save_pretrained(pt_save_directory)