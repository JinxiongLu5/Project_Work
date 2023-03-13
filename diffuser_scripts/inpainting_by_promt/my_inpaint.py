import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline
import cv2
import numpy as np 
import time, os
import xml.etree.ElementTree as ET

num_init = 1
num_mask = 1
num_prompt = 1
num_img_per = 1
count = 0

root_path = "data/voc/"

with open('prompts.txt') as prompts_file:
    prompts_list = prompts_file.readlines()
device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

for init_label in range(num_init):
    # order init_image from initial_img_with_ROI Folder
    init_path = 'initial_img_with_ROI/'
    dir_list_i = os.listdir(init_path)
    image_path = init_path+dir_list_i[init_label]
    init_image = cv2.imread(image_path)
    init_image = PIL.Image.fromarray(init_image.squeeze(), mode="RGB").convert("RGB").resize((512, 512))
    for _ in range(num_mask):
        # sample mask_image from mask_img Folder
        mask_path = "mask_img/" + str(init_label) + "/"
        m_img_list = os.listdir(mask_path)
        idx = np.random.randint(len(m_img_list))
        image_path = mask_path+m_img_list[idx]
        mask_image = cv2.imread(image_path)
        mask_image = PIL.Image.fromarray(mask_image.squeeze(), mode="RGB").convert("RGB").resize((512, 512))
        for _ in range(num_prompt):
            # sample from prompts.txt 
            prompt = prompts_list[np.random.randint(len(prompts_list))].strip('\n')
            images = pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                guidance_scale=7.5,
                generator=torch.Generator(device="cuda").manual_seed(count),
                num_images_per_prompt=num_img_per,
            ).images
            for i in range(num_img_per):
                img_name = time.strftime("%Y%m%d-%H%M%S")+"-"+str(count)+ "-"+ m_img_list[idx].strip('.png')
                image_str = root_path+ 'JPEGImages/' +img_name+'.png'
                images[i].save(image_str)
                count += 1 
                (ymin, ymax, xmin, xmax) = m_img_list[idx].strip('.png').split('-')[1].split('_')[0:4]
                tree = ET.parse('data/empty_annotation.xml')
                root = tree.getroot()
                obj = ET.parse('data/empty_object.xml')
                obj_root = obj.getroot()
                root[1].text = img_name+'.png' # filename
                # append label
                obj_root[0].text = prompt.split(" ")[1] # second word as the class_name
                obj_root[4][1].text = ymin
                obj_root[4][3].text = ymax
                obj_root[4][0].text = xmin
                obj_root[4][2].text = xmax
                root.append(obj_root)
                tree.write(root_path+ 'Annotations/'+img_name+'.xml')
    
    
