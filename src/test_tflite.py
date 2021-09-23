import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

tflite_model_path = "/home/kaushikdas/aashish/multi-mnist/saved_models/worksheet.multi_digit_zone_clf_transformer_attempt_1.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

base_path = "/home/kaushikdas/aashish/multi-digit-dataset/output_comma_2/output_mix/test/combined_1_to_8_comma_actual_shape_real/"
img_name = "360000.png"
orig_img = cv2.imread(base_path + img_name, 0)

img = orig_img

img_height = img.shape[0]
img_width = img.shape[1]
img = img.reshape((1,img_height, img_width, 1))
print(img.shape)

img = np.transpose(img, axes=[0,2,1,3])
img = img / 255
print(img.shape)

img = img.astype('float32')
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]['index'], (1, img.shape[1], img.shape[2], 1), strict=True)
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)
print(np.argmax(output_data))