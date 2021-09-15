import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from shutil import copyfile
import albumentations as A
import argparse


from multi_digit_dataloader import MultiDigitDataLoaderActualShape
from utils import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-model_path", help = "Path of model", type=str, default=None, required=True)
    parser.add_argument("-prefix", help = "Prefix folder name for output", type=str, default=None, required=True)

    args = parser.parse_args()
    return args

def get_preds(dataset, model):
    pred_texts = []
    for i, batch in enumerate(dataset):
        if i%1000==0:
            print(i)
        batch_images = batch["image"]
        preds = model.predict(batch_images)
        pred_texts_temp = decode_batch_predictions(preds)
        pred_texts += pred_texts_temp
    return pred_texts

def main():
    args = parse_args()
    figure_folder_prefix = figure_folder + args.prefix
    test_figure_folder_prefix = figure_folder_prefix + "/test_actual_shape"
    if not os.path.isdir(test_figure_folder_prefix):
        os.mkdir(test_figure_folder_prefix)

    model = tf.keras.models.load_model(args.model_path)
    print(model.summary())

    # batch_size = 1
    # num_time_steps = 42
    # max_digit_length = 8
    shuffle = False

    test_transform = A.Compose([
                                A.ToFloat(max_value=255)
                               ])
    
    test_multi_digit_dataset = MultiDigitDataLoaderActualShape(df_path = test_df_path,
                                                               image_base_path = test_image_actual_shape_base_path,
                                                               num_time_steps = num_time_steps,
                                                               transform = test_transform,
                                                               max_digit_length = max_digit_length,
                                                               shuffle = shuffle)
    
    print("test_multi_digit_dataset length: ",len(test_multi_digit_dataset))
    print("\n\nSanity Check Testing Data")
    check_dataset(test_multi_digit_dataset)

    print("\nCalculating Predictions\n")

    test_pred_texts = get_preds(test_multi_digit_dataset, model)

    df_test_labels = pd.read_csv(test_df_path, header = None, dtype={0: str, 1: str})
    df_test_labels.columns = ["file_name", "labels"]
    df_test_labels['preds'] = test_pred_texts

    print("\nCalculating Edit Distance\n")
    test_edit_distance_freq, test_wrong_count = get_edit_distance_freq(df_test_labels)
    show_edit_distance_freq_graph(test_edit_distance_freq, "test_edit_distance_frequency", test_figure_folder_prefix)


    print("\nCalculating Word Length Frequency\n")
    test_correct_word_length_freq, test_wrong_word_length_freq = get_word_leng_freq(df_test_labels)
    show_word_length_freq_graph(test_wrong_word_length_freq,"test_wrong_word_length_frequency", test_figure_folder_prefix)
    show_word_length_freq_graph(test_correct_word_length_freq,"test_correct_word_length_frequency", test_figure_folder_prefix)

    total_test_samples = df_test_labels.shape[0]
    print("Test wrong:", test_wrong_count)
    print("Test size:", total_test_samples)
    print("Test Error %:",test_wrong_count/total_test_samples*100)
    print("Test Accuracy %:",(1 - (test_wrong_count/total_test_samples))*100)

    print("Correct percentage")
    for key in test_wrong_word_length_freq:
        total_sample = test_wrong_word_length_freq[key] + test_correct_word_length_freq[key]
        correct_percentage = (test_correct_word_length_freq[key]/total_sample)*100
        print("{} {:.2f}% ({}/{})".format(key,correct_percentage,test_wrong_word_length_freq[key],total_sample))

    test_accuracy_file = open(test_figure_folder_prefix + '/test_accuracy.txt', 'w')
    test_accuracy_file.write("Test wrong: {}\n".format(test_wrong_count))
    test_accuracy_file.write("Test size: {}\n".format(total_test_samples))
    test_accuracy_file.write("Test Error %: {}\n".format(test_wrong_count/total_test_samples*100))
    test_accuracy_file.write("Test Accuracy %: {:.2f}\n".format((1 - (test_wrong_count/total_test_samples))*100))
    test_accuracy_file.write("Correct percentage\n")
    for key in test_wrong_word_length_freq:
        total_sample = test_wrong_word_length_freq[key] + test_correct_word_length_freq[key]
        correct_percentage = (test_correct_word_length_freq[key]/total_sample)*100
        test_accuracy_file.write("{} {:.2f}% ({}/{})\n".format(key,correct_percentage,test_wrong_word_length_freq[key],total_sample))
    test_accuracy_file.close()


if __name__ == '__main__':
    main()