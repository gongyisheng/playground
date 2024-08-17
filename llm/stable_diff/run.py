import torch
from diffusers import StableDiffusion3Pipeline
import time

pipe = StableDiffusion3Pipeline.from_single_file(
    "/home/yisheng/stable-diffusion-3-medium/sd3_medium_incl_clips.safetensors", 
    text_encoder_3=None,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe(
    "images of woman with her owls, in the style of ethereal beauty, porcelain, dark white and light azure, clifford coffin, northern renaissance, white and amber, wiccan --ar 2:3 --v 6.0",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    height=1024,
    width=1024,
).images[0]

image.save(f"{int(time.time())}.png")