import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from shutil import copyfile


train_folder = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/real_data_hw/set-3/train"
save_img_folder = os.path.join(train_folder, "all")
if not os.path.isdir(save_img_folder):
    os.mkdir(save_img_folder)
print(save_img_folder)

data_path = "/Users/aashishkumar/Documents/Projects/vision/training_database/worksheet_zone/recognize_multi_digits/training/"
list_subfolders_with_paths = []
for f in os.scandir(data_path):
    if f.is_dir():
        if f.name == "-5" or f.name == "-4" or f.name == "-3" or f.name == "-2" or f.name == "-1":
            continue
        list_subfolders_with_paths.append(f.path)


file_out = open(os.path.join(train_folder, 'labels.csv'), 'w+')
img_name_counter = 0
for folder in list_subfolders_with_paths:
    image_path_list = glob.glob(folder + "/*.bmp")
    label = folder.split('/')[-1]
    label = "".join(['a' if x==',' else x for x in label])
    for img_path in image_path_list:
        if img_name_counter%100 == 0:
            print(img_name_counter)
        img_name = str(img_name_counter)
        output_img_name = img_name + ".bmp"
        output_img_path = save_img_folder + "/" + output_img_name
        copyfile(img_path, output_img_path)
        file_out.write(img_name + '.bmp,' + label + '\n')
        img_name_counter += 1

file_out.close()
print(img_name_counter)
print("Done")