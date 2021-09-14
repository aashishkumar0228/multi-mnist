import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random

random.seed(1)


def modify_image_old(image_path):
    original_img = cv2.imread(image_path, 0)
    img = original_img.copy()
    for i in range(0, original_img.shape[0]):
        for j in range(0, original_img.shape[1]):
            if original_img[i][j] == 255:
                img[i][j] = random.randrange(190,210)
    blurImg = cv2.blur(img,(3,3))
    return blurImg


# base_path      = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_comma_1/test/"
# new_img_folder = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_comma_1/test/"

base_path      = "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_1/train/"
new_img_folder = "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_1/train/"

old_folder = ['combined_4_to_8_random']
new_folder = ['combined_4_to_8_random_real']
for digit, digit_new in zip(old_folder,new_folder):
    print(digit)
    image_path_list = glob.glob(base_path + digit + "/*.png")
    print('total images: ', len(image_path_list))
    if not os.path.isdir(new_img_folder + digit_new):
        os.mkdir(new_img_folder + digit_new)
    for i in tqdm(range(len(image_path_list))):
        image_path = image_path_list[i]
        img = modify_image_old(image_path)
        img_name = image_path.split("/")[-1]
        new_img_path = new_img_folder + digit_new + "/" + img_name
        cv2.imwrite(new_img_path, img)
    

print("Done")
