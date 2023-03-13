import PIL
import torch
from diffusers import DiffusionPipeline
import cv2
import numpy as np 
import time, os
import xml.etree.ElementTree as ET

num_init = 1
num_img = 1
max_num_items = 1

seed_count = 0
count = 0
root_path = "data/voc/"
 
pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
).to("cuda")

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
            # sample exam_img
            folder_idx = np.random.randint(len(os.listdir("exam_img/")))
            folder_name = os.listdir("exam_img/")[folder_idx]
            exam_path = "exam_img/" + folder_name
            e_idx = np.random.randint(len(os.listdir(exam_path)))
            e_image_path = os.path.join(exam_path, os.listdir(exam_path)[e_idx])
            example_image = cv2.imread(e_image_path)
            example_image = PIL.Image.fromarray(example_image.squeeze(), mode="RGB").convert("RGB").resize((512, 512))
            images = pipe(image=init_image,\
             mask_image=mask_image, \
             example_image=example_image\
             ).images
            seed_count += 1 
            obj = ET.parse('data/empty_object.xml')
            obj_root = obj.getroot()
            # append label
            obj_root[0].text = folder_name # folder name as the class_name
            (ymin, ymax, xmin, xmax) = m_img_list[idx].strip('.png').split('-')[1].split('_')[0:4]
            obj_root[4][1].text = ymin
            obj_root[4][3].text = ymax
            obj_root[4][0].text = xmin
            obj_root[4][2].text = xmax
            root.append(obj_root)

        name = time.strftime("%Y%m%d-%H%M%S")+"-"+str(count)+ "-"+ str(num_items) +"_num_items"
        image_str = root_path+ 'JPEGImages/' +name+'.png'
        root[1].text = name+'.png' # filename
        # images[0].save(image_str)
        open_cv_image = np.array(images[0]) 
        cv2.imwrite(image_str, open_cv_image)
        tree.write(root_path+ 'Annotations/'+name+'.xml')
        count += 1 
        
        
        
