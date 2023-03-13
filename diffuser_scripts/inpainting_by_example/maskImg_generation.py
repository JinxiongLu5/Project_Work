import cv2
import numpy as np
import sys, os

num_masks = 1
img_height, img_width, img_channel = 512, 512, 3
ori_bbox_height, ori_bbox_width = 150, 150
len_var = 30
path = 'initial_img_with_ROI/'
dir_list = os.listdir(path)
for i, dir in enumerate(dir_list):
    items = dir.split('_')
    ROIymin, ROIymax, ROIxmin, ROIxmax = int(items[1]), int(items[2]), int(items[3]), int(items[4]) 
    folder_path = "mask_img/"+ str(i)+"/"
    os.system("rm -rf {0}".format(folder_path))
    os.makedirs(folder_path)
    for j in range(num_masks):
        ran_y = np.random.randint(ROIymin, ROIymax-ori_bbox_height-len_var)
        ran_x = np.random.randint(ROIxmin, ROIxmax-ori_bbox_width-len_var)
        ran_height= np.random.randint(ori_bbox_height, ori_bbox_height+len_var)
        ran_width= np.random.randint(ori_bbox_width, ori_bbox_width+len_var)
        mask_image = np.zeros((img_height, img_width, img_channel))
        mask_image[ran_y:(ran_y+ran_height),ran_x:(ran_x+ran_width), :] = 255
        lable = str(ran_y) + "_" + str(ran_y+ran_height) + "_" + str(ran_x) + "_" + str(ran_x+ran_width) + "_"
        cv2.imwrite(folder_path+str(j)+"-"+lable+".png", mask_image)
