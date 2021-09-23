import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


model_path = "/home/kaushikdas/aashish/multi-mnist/saved_models/transformer_attempt_1/multi_digits_zone_classifier"
model_tflite_name = "/home/kaushikdas/aashish/multi-mnist/saved_models/transformer_attempt_1/worksheet.multi_digit_zone_transformer_attempt_tf_2.4.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
# Save the TF Lite model.
with tf.io.gfile.GFile(model_tflite_name, 'wb') as f:
    f.write(tflite_model)