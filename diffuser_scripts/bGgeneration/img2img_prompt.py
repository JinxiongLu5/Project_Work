import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import cv2
import numpy as np 
import time, os
import PIL

# load the pipeline
device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

init_path = 'cropped_img/'
dir_list_i = os.listdir(init_path)
num_bg = len(dir_list_i)

for n_th_bg in range(num_bg):
    image_path = init_path+dir_list_i[n_th_bg]
    init_image = cv2.imread(image_path) 
    init_image = PIL.Image.fromarray(init_image.squeeze(), mode="RGB").convert("RGB").resize((768, 512))
    prompt = "similar background"
    images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    img_name = 'prompt_generated_background/b'+str(n_th_bg)+'.png'
    images[0].save(img_name)