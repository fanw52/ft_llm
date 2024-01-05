import os

import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image

from huggingface_hub import snapshot_download
from torch import autocast

os.environ['HTTP_PROXY'] = "http://192.168.51.143:10811"
os.environ['HTTPS_PROXY'] = "http://192.168.51.143:10811"
os.environ["no_proxy"] = "localhost, 127.0.0.1"

# IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1
# runwayml/stable-diffusion-v1-5
# /data/pretrained_models/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1
# model_path = "runwayml/stable-diffusion-v1-5"
# snapshot_download(repo_id=model_path,
#                   local_dir="/data/pretrained_models/stable-diffusion-v1-5")

# /data/pretrained_models/stable-diffusion-v1-5
pipe = AutoPipelineForText2Image.from_pretrained("/data1/pretrained_models/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# pipeline = DiffusionPipeline.from_pretrained("/data/pretrained_models/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1", torch_dtype=torch.float16)
pipe.to("cuda")


def generate(prompt="An image of a squirrel in Picasso style"):
    with autocast("cuda"):
        # image = pipe(
        #     prompt,
        #     height=512,
        #     width=512,
        #     num_inference_steps=150,
        #     guidance_scale=7.5,
        #     negative_prompt=None,
        #     num_images_per_prompt=1,
        # )["images"][0]
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    return image


demo = gr.Interface(generate, gr.Text(), "image")
demo.launch(server_name='0.0.0.0', server_port=8080, show_error=True)
