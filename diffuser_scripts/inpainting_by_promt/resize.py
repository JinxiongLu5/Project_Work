import cv2
import numpy as np
import sys, os

path = 'un_resize_img/'
dir_list = os.listdir(path)
desired_height = 512
desired_width = 512
for i in range(len(dir_list)):
    image_path = path+dir_list[i]
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (desired_height, desired_width))
    cv2.imwrite("resized_img/"+ str(i)+".png", resized_img)
    # cv2.imshow("Resized image", resized_img)
    # cv2.waitKey(0)
cv2.destroyAllWindows()