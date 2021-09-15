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
num_samples_digits = [4000//2, 5000//2, 6000//2, 7000//2, 8000//2]

# num_digits = [1]
# num_samples_digits = [1000]
max_width = 168
max_height = 28


base_path =         "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_comma_2/test/"
output_base_path =  "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/output_comma_2/test/combined_4_to_8_actual_shape/"
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
        output_img_name = img_name[:-4] + ".png"
        output_img_path = output_base_path + output_img_name
        copyfile(img_path, output_img_path)
        
print(k)
print("done")