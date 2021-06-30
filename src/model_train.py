import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A

from multi_digit_dataloader import MultiDigitDataLoader
from models import build_model_small
from utils import *
from config import *


if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if not os.path.isdir(figure_folder):
    os.mkdir(figure_folder)

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
model = build_model_small(img_height)
print(model.summary())
print("\n\n")

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history_1 = model.fit(
                      train_multi_digit_dataset, 
                      validation_data=test_multi_digit_dataset,
                      epochs=epochs, 
                      callbacks=[early_stopping]
                      )

get_loss_plot(history_1, figure_folder)
loss_df = pd.DataFrame(list(zip(history_1.epoch, history_1.history['loss'], history_1.history['val_loss'])), 
                        columns = ['Epoch', 'Train_Loss', 'Val_loss'])
loss_df.to_csv(figure_folder + "loss.csv", index=None)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
print("\n\n")
print(prediction_model.summary())
print("\n\n")

pred_texts = []
for batch in train_multi_digit_dataset:
    batch_images = batch["image"]
    preds = prediction_model.predict(batch_images)
    pred_texts_temp = decode_batch_predictions(preds)
    pred_texts += pred_texts_temp

df_train_labels = train_multi_digit_dataset.df.copy()
df_train_labels['preds'] = pred_texts

print("\nCalculating Edit Distance\n")
train_edit_distance_freq, train_wrong_count = get_edit_distance_freq(df_train_labels)
show_edit_distance_freq_graph(train_edit_distance_freq, "train_edit_distance_frequency", figure_folder)



print("\nCalculating Word Length Frequency\n")
train_correct_word_length_freq, train_wrong_word_length_freq = get_word_leng_freq(df_train_labels)
show_word_length_freq_graph(train_wrong_word_length_freq,"train_wrong_word_length_frequency", figure_folder)


# save prediction model
print("\n\nSaving Prediction Model\n\n")
prediction_model.save(model_folder_name)

model_json = prediction_model.to_json()
with open(model_json_file_name, "w") as json_file:
    json_file.write(model_json)

prediction_model.save_weights(model_weights_file_name)


# convert to tflite model
print("\n\nConverting to TF Lite \n\n")
converter = tf.lite.TFLiteConverter.from_keras_model(prediction_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with tf.io.gfile.GFile(model_tflite_name, 'wb') as f:
    f.write(tflite_model)


total_train_samples = df_train_labels.shape[0]
print("Train wrong:", train_wrong_count)
print("Train size:", total_train_samples)
print("Train Error %:",train_wrong_count/total_train_samples*100)
print("Train Accuracy %:",(1 - (train_wrong_count/total_train_samples))*100)
