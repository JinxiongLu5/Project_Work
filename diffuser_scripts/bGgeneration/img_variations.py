from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import cv2
import numpy as np 
import time, os
import PIL
from torchvision import transforms

device = "cuda"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

init_path = 'cropped_img/'
dir_list_i = os.listdir(init_path)
num_bg = len(dir_list_i)
for n_th_bg in range(num_bg):
    image_path = init_path+dir_list_i[n_th_bg]
    # init_image = cv2.imread(image_path) 
    # init_image = PIL.Image.fromarray(init_image.squeeze(), mode="RGB").convert("RGB").resize((224, 224))
    init_image = Image.open(image_path)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(init_image).to(device) 
    out = sd_pipe(inp, guidance_scale=3)
    img_name = 'var_generated_background/b'+str(n_th_bg)+'.png' 
    out["images"][0].save(img_name)
