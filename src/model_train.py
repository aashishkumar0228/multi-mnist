import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from multi_digit_datagen import MultiDigitGenerator
from models import build_model_small
from utils import *


# model names to save
model_folder_name = "multi_digit_model_1_to_5_real"
model_json_file_name = "multi_digit_model_1_to_5_real_json.json"
model_weights_file_name = "multi_digit_model_1_to_5_real_weights.h5"
model_tflite_name = "worksheet.multi_digit_model_1_to_5_real.tflite"

# dataset paths
train_df_path = "/home/kaushikdas/aashish/multi-digit-dataset/train/labels_1_to_8.csv"
train_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/train/combined_1_to_8_real/"
test_df_path = "/home/kaushikdas/aashish/multi-digit-dataset/test/labels_1_to_8.csv"
test_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/test/combined_1_to_8_real/"

batch_size = 32
img_height = 28
img_width = 168
num_time_steps = 42  # img_width//4
max_digit_length = 8
shuffle = True

epochs = 1
early_stopping_patience = 3

# get train and test dataset
train_multi_digit_dataset = MultiDigitGenerator(df_path = train_df_path,
                                                image_base_path = train_image_base_path,
                                                batch_size = batch_size,
                                                img_height = img_height,
                                                img_width = img_width,
                                                num_time_steps = num_time_steps,
                                                max_digit_length = max_digit_length,
                                                shuffle = shuffle)

test_multi_digit_dataset = MultiDigitGenerator(df_path = test_df_path,
                                               image_base_path = test_image_base_path,
                                               batch_size = batch_size,
                                               img_height = img_height,
                                               img_width = img_width,
                                               num_time_steps = num_time_steps,
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

plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set
plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('loss.png')

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
train_edit_distance_freq, test_wrong_count = get_edit_distance_freq(df_train_labels)
show_edit_distance_freq_graph(train_edit_distance_freq, "train_edit_distance_frequency")

total_train_samples = df_train_labels.shape[0]
print("Train wrong:", test_wrong_count)
print("Train size:", total_train_samples)
print("Train Error %:",test_wrong_count/total_train_samples*100)
print("Train Accuracy %:",(1 - (test_wrong_count/total_train_samples))*100)

print("\nCalculating Word Length Frequency\n")
train_correct_word_length_freq, train_wrong_word_length_freq = get_word_leng_freq(df_train_labels)
show_word_length_freq_graph(train_wrong_word_length_freq,"train_wrong_word_length_frequency")


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
