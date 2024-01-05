import gradio as gr
import torch
from diffusers import DiffusionPipeline

# load both base & refiner
base_model_path = "/data1/pretrained_models/stable-diffusion-xl-base-1.0"
refiner_model_path = "/data1/pretrained_models/stable-diffusion-xl-refiner-1.0"


base = DiffusionPipeline.from_pretrained(
    base_model_path, torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    refiner_model_path,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"

def generate(prompt="An image of a squirrel in Picasso style"):
    # image = pipe(prompt=prompt, num_inference_steps=100, guidance_scale=0.0).images[0]
    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    # image.save("demo2.jpg")
    return image


demo = gr.Interface(generate, gr.Text(), "image")
demo.launch(server_name='0.0.0.0', show_error=True)
