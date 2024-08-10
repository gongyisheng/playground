import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_single_file(
    "/home/yisheng/stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp16.safetensors", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    height=1024,
    width=1024,
).images[0]

image.save("ssd3_hello_world.png")