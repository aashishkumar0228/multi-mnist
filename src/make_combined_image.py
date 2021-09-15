import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from random import randrange
from shutil import copyfile


# num_digits = [1, 2, 3, 4, 5, 6, 7, 8]
# num_samples_digits = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
# num_samples_digits = [ 1000,  2000,  3000,  4000,  5000,  6000,  7000,  8000]
num_digits = [4, 5, 6, 7, 8]
num_samples_digits = [40000//2, 50000//2, 60000//2, 70000//2, 80000//2]

# num_digits = [1]
# num_samples_digits = [1000]
max_width = 168
max_height = 28

base_path =         "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_comma_2/train/"
output_base_path =  "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_comma_2/train/combined_4_to_8_random/"
if not os.path.isdir(output_base_path):
    os.mkdir(output_base_path)

labels_csv_path = base_path + "labels_4_to_8.csv"
df_labels = pd.read_csv(labels_csv_path, header = None, dtype={0: str, 1: str})
print(df_labels.head())

k = -1
for digit, num_samples in zip(num_digits, num_samples_digits):
    print("digit",digit)
    for i in range(num_samples):
        k += 1
        if k%1000 == 0:
            print(k)
        img_name = df_labels.iloc[k][0]
        label = df_labels.iloc[k][1]
        img_path = base_path + str(digit) + "/" + img_name
        # print(img_path)
        img = cv2.imread(img_path, 0)
        if img.shape[1] > max_width:
            final_img = cv2.resize(img, (max_width, max_height))
        else:
            final_img = np.ones([max_height, max_width], dtype='uint8')*255
            diff_width = final_img.shape[1] - img.shape[1]
            start_x = randrange(diff_width+1)
            final_img[:,start_x : start_x + img.shape[1]] = img
        
        output_img_name = img_name[:-4] + ".png"
        output_img_path = output_base_path + output_img_name
        cv2.imwrite(output_img_path, final_img)
        
print(k)
print("done")


