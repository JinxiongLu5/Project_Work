import cv2
import numpy as np
import sys, os

path = 'data_background/'
dir_list = os.listdir(path)
# desired_height = 512
# desired_width = 512
for i in range(len(dir_list)):
    image_path = path+dir_list[i]
    img = cv2.imread(image_path)
    height, width, channel = img.shape[:]
    img_crop = img[:int(height/2), :, :]
    # resized_img = cv2.resize(img, (desired_height, desired_width))
    cv2.imwrite("cropped_img/"+ str(i)+".png", img_crop)
    # cv2.imshow("Resized image", resized_img)
    # cv2.waitKey(0)
cv2.destroyAllWindows()