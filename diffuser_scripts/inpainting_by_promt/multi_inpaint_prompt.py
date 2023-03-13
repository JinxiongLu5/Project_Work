import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline
import cv2
import numpy as np 
import time, os
import xml.etree.ElementTree as ET

num_init = 2
num_img = 50
max_num_items = 5

seed_count = 0
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
    
    for _ in range(num_img):
        tree = ET.parse('data/empty_annotation.xml')
        root = tree.getroot()
        num_items = np.random.randint(1, 1+max_num_items)

        for n in range(num_items):
            input_image = init_image if n==0 else images[0]
            # sample mask_image from mask_img Folder
            mask_path = "mask_img/" + str(init_label) + "/"
            m_img_list = os.listdir(mask_path)
            idx = np.random.randint(len(m_img_list))
            image_path = mask_path+m_img_list[idx]
            mask_image = cv2.imread(image_path)
            mask_image = PIL.Image.fromarray(mask_image.squeeze(), mode="RGB").convert("RGB").resize((512, 512))
            # sample from prompts.txt 
            prompt = prompts_list[np.random.randint(len(prompts_list))].strip('\n')
            images = pipe(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image,
                guidance_scale=7.5,
                generator=torch.Generator(device="cuda").manual_seed(count),
                num_images_per_prompt=1,
            ).images
            seed_count += 1 
            obj = ET.parse('data/empty_object.xml')
            obj_root = obj.getroot()
            # append label
            obj_root[0].text = prompt.split(" ")[1] # second word as the class_name
            (ymin, ymax, xmin, xmax) = m_img_list[idx].strip('.png').split('-')[1].split('_')[0:4]
            obj_root[4][1].text = ymin
            obj_root[4][3].text = ymax
            obj_root[4][0].text = xmin
            obj_root[4][2].text = xmax
            root.append(obj_root)

        name = time.strftime("%Y%m%d-%H%M%S")+"-"+str(count)+ "-"+ str(num_items) +"_num_items"
        image_str = root_path+ 'JPEGImages/' +name+'.png'
        root[1].text = name+'.png' # filename
        images[0].save(image_str)
        tree.write(root_path+ 'Annotations/'+name+'.xml')
        count += 1 
        
        
        
