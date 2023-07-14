import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

model_id = "runwayml/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,  # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

# 定义参数
prompt = [
    "A dragon fruit wearing karate belt in the snow",
    "A small cactus wearing a straw hat and neon sunglasses in the Sahara desert",
    "A photo of a raccoon wearing an astronaut helmet, looking out of the window at night",
    "A cute otter in a rainbow whirlpool holding shells, watercolor"
]
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 7.5
negative_prompt = ""
batch_size = len(prompt)
# 随机种子
generator = torch.Generator(device).manual_seed(2023)

with torch.no_grad():
    # 获取text_embeddings
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
# 获取unconditional text embeddings
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
# 拼接为batch，方便并行计算
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# 生成latents的初始噪音
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator, device=device
)
latents = latents.to(device, dtype=torch.float16)

# 设置采样步数
noise_scheduler.set_timesteps(num_inference_steps, device=device)

# scale the initial noise by the standard deviation required by the scheduler
latents = latents * noise_scheduler.init_noise_sigma  # for DDIM, init_noise_sigma = 1.0

timesteps_tensor = noise_scheduler.timesteps

# Do denoise steps
for t in tqdm(timesteps_tensor):
    # 这里latens扩展2份，是为了同时计算unconditional prediction
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)  # for DDIM, do nothing

    # 使用UNet预测噪音
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

# 执行CFG
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

# 计算上一步的noisy latents：x_t -> x_t-1
latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

# 注意要对latents进行scale
latents = 1 / 0.18215 * latents
# 使用vae解码得到图像
image = vae.decode(latents).sample
#
# import torch
# from diffusers import StableDiffusionPipeline
# from PIL import Image
#
#
# # 组合图像，生成grid
# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows * cols
#
#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols * w, rows * h))
#     grid_w, grid_h = grid.size
#
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i % cols * w, i // cols * h))
#     return grid
#
#
# # 加载文生图pipeline
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",  # 或者使用 SD v1.4: "CompVis/stable-diffusion-v1-4"
#     torch_dtype=torch.float16
# ).to("cuda")
#
# # 输入text，这里text又称为prompt
# prompts = [
#     "a photograph of an astronaut riding a horse",
#     "A cute otter in a rainbow whirlpool holding shells, watercolor",
#     "An avocado armchair",
#     "A white dog wearing sunglasses"
# ]
#
# generator = torch.Generator("cuda").manual_seed(42)  # 定义随机seed，保证可重复性
#
# # 执行推理
# images = pipe(
#     prompts,
#     height=512,
#     width=512,
#     num_inference_steps=50,
#     guidance_scale=7.5, # guidance_scale越大时，生成的图像应该会和输入文本更一致，并不是越大越好
#     negative_prompt=None, # 使用不为空的negative_prompt来避免模型生成的图像包含不想要的东西
#     num_images_per_prompt=1,
#     generator=generator
# ).images
#
# grid = image_grid(images, rows=1, cols=4)
