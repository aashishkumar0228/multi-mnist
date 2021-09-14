import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A
import argparse
import time
import tensorflow_addons as tfa



from multi_digit_dataloader import MultiDigitDataLoader
from models import build_big_model_no_compile
from utils import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-prefix", help = "Prefix folder name for output", type=str, default=None)
    args = parser.parse_args()
    return args

def get_preds(dataset, model):
    pred_texts = []
    for batch in dataset:
        batch_images = batch["image"]
        preds = model.predict(batch_images)
        pred_texts_temp = decode_batch_predictions(preds)
        pred_texts += pred_texts_temp
    return pred_texts

def main():
    args = parse_args()
    prefix = args.prefix
    if prefix is None:
        ts = int(time.time())
        prefix = str(ts)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(figure_folder):
        os.mkdir(figure_folder)
    
    output_dir_prefix = output_dir + prefix
    figure_folder_prefix = figure_folder + prefix

    if not os.path.isdir(output_dir_prefix):
        os.mkdir(output_dir_prefix)
    if not os.path.isdir(figure_folder_prefix):
        os.mkdir(figure_folder_prefix)

    train_transform = A.Compose([
                                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.1, p=1),
                                A.ShiftScaleRotate(shift_limit_x=0.01, shift_limit_y=0.1, rotate_limit=5),
                                A.ToFloat(max_value=255)
                            ])

    test_transform = A.Compose([
                                A.ToFloat(max_value=255)
                            ])

    # get train and test dataset
    train_multi_digit_dataset = MultiDigitDataLoader(df_path = train_df_path,
                                                    image_base_path = train_image_base_path,
                                                    batch_size = batch_size,
                                                    img_height = img_height,
                                                    img_width = img_width,
                                                    num_time_steps = num_time_steps,
                                                    transform = train_transform,
                                                    max_digit_length = max_digit_length,
                                                    shuffle = shuffle)

    test_multi_digit_dataset = MultiDigitDataLoader(df_path = test_df_path,
                                                    image_base_path = test_image_base_path,
                                                    batch_size = batch_size,
                                                    img_height = img_height,
                                                    img_width = img_width,
                                                    num_time_steps = num_time_steps,
                                                    transform = test_transform,
                                                    max_digit_length = max_digit_length,
                                                    shuffle = shuffle)

    print("Sanity Check Training Data")
    check_dataset(train_multi_digit_dataset)
    print("\n\nSanity Check Testing Data")
    check_dataset(test_multi_digit_dataset)
    print("\n\n")

    # Get the model
    model = build_big_model_no_compile(img_height, num_classes)
    opt = keras.optimizers.Adam()
    # moving_avg_opt = tfa.optimizers.MovingAverage(opt)
    stochastic_avg_opt = tfa.optimizers.SWA(opt)
    # stochastic_avg_opt = tfa.optimizers.SWA(opt, start_averaging=4, average_period=5)

    # model.compile(optimizer=moving_avg_opt)
    model.compile(optimizer=stochastic_avg_opt)

    print(model.summary())
    print("\n\n")

    # Add early stopping
    # early_stopping = keras.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    # )
    # reduce learning rate
    lrr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,verbose=1,factor=0.5, min_lr=0.00001)

    # Train the model
    history_1 = model.fit(
                        train_multi_digit_dataset, 
                        validation_data=test_multi_digit_dataset,
                        epochs=epochs, 
                        callbacks=[lrr]
                        )

    get_loss_plot(history_1, figure_folder_prefix)
    create_loss_csv(history_1, figure_folder_prefix)

    # get average weights
    print("\n\nAveraging layer weights\n\n")
    # moving_avg_opt.assign_average_vars(model.variables)
    stochastic_avg_opt.assign_average_vars(model.variables)

    print("\n\nSaving Original Model\n\n")
    model.save(output_dir_prefix + "/" + model_folder_name + "_original")

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    print("\n\n")
    print(prediction_model.summary())
    print("\n\n")
    
    train_pred_texts = get_preds(train_multi_digit_dataset, prediction_model)
    df_train_labels = train_multi_digit_dataset.df.copy()
    df_train_labels['preds'] = train_pred_texts

    test_pred_texts = get_preds(test_multi_digit_dataset, prediction_model)
    df_test_labels = test_multi_digit_dataset.df.copy()
    df_test_labels['preds'] = test_pred_texts

    print("\nCalculating Edit Distance\n")
    train_edit_distance_freq, train_wrong_count = get_edit_distance_freq(df_train_labels)
    show_edit_distance_freq_graph(train_edit_distance_freq, "train_edit_distance_frequency", figure_folder_prefix)

    test_edit_distance_freq, test_wrong_count = get_edit_distance_freq(df_test_labels)
    show_edit_distance_freq_graph(test_edit_distance_freq, "test_edit_distance_frequency", figure_folder_prefix)

    print("\nCalculating Word Length Frequency\n")
    train_correct_word_length_freq, train_wrong_word_length_freq = get_word_leng_freq(df_train_labels)
    show_word_length_freq_graph(train_wrong_word_length_freq,"train_wrong_word_length_frequency", figure_folder_prefix)
    show_word_length_freq_graph(train_correct_word_length_freq,"train_correct_word_length_frequency", figure_folder_prefix)

    test_correct_word_length_freq, test_wrong_word_length_freq = get_word_leng_freq(df_test_labels)
    show_word_length_freq_graph(test_wrong_word_length_freq,"test_wrong_word_length_frequency", figure_folder_prefix)
    show_word_length_freq_graph(test_correct_word_length_freq,"test_correct_word_length_frequency", figure_folder_prefix)

    total_train_samples = df_train_labels.shape[0]
    print("Train wrong:", train_wrong_count)
    print("Train size:", total_train_samples)
    print("Train Error %:",train_wrong_count/total_train_samples*100)
    print("Train Accuracy %:",(1 - (train_wrong_count/total_train_samples))*100)

    total_test_samples = df_test_labels.shape[0]
    print("Test wrong:", test_wrong_count)
    print("Test size:", total_test_samples)
    print("Test Error %:",test_wrong_count/total_test_samples*100)
    print("Test Accuracy %:",(1 - (test_wrong_count/total_test_samples))*100)
    
    train_accuracy_file = open(figure_folder_prefix + '/train_test_accuracy.txt', 'w')
    train_accuracy_file.write("Train wrong: {}\n".format(train_wrong_count))
    train_accuracy_file.write("Train size: {}\n".format(total_train_samples))
    train_accuracy_file.write("Train Error %: {}\n".format(train_wrong_count/total_train_samples*100))
    train_accuracy_file.write("Train Accuracy %: {:.2f}\n\n".format((1 - (train_wrong_count/total_train_samples))*100))
    train_accuracy_file.write("Test wrong: {}\n".format(test_wrong_count))
    train_accuracy_file.write("Test size: {}\n".format(total_test_samples))
    train_accuracy_file.write("Test Error %: {}\n".format(test_wrong_count/total_test_samples*100))
    train_accuracy_file.write("Test Accuracy %: {:.2f}\n".format((1 - (test_wrong_count/total_test_samples))*100))
    train_accuracy_file.close()

    # save prediction model
    print("\n\nSaving Prediction Model\n\n")
    prediction_model.save(output_dir_prefix + "/" + model_folder_name)

    model_json = prediction_model.to_json()
    with open(output_dir_prefix + "/" + model_json_file_name, "w") as json_file:
        json_file.write(model_json)

    prediction_model.save_weights(output_dir_prefix + "/" + model_weights_file_name)

    # convert to tflite model
    print("\n\nConverting to TF Lite \n\n")
    converter = tf.lite.TFLiteConverter.from_keras_model(prediction_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(output_dir_prefix + "/" + model_tflite_name, 'wb') as f:
        f.write(tflite_model)
    
    print("Done")


    


if __name__ == '__main__':
    main()