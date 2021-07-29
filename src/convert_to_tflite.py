import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


model_path = "/home/kaushikdas/aashish/multi-mnist/saved_models/big_model_random_scale_stochastic_average/multi_digit_model_1_to_8_real"



model_tflite_name = "worksheet.multi_digit_model_1_to_8_real_big_model_random_scale_stochastic_average_2.tflite"



converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
# Save the TF Lite model.
with tf.io.gfile.GFile(model_tflite_name, 'wb') as f:
    f.write(tflite_model)