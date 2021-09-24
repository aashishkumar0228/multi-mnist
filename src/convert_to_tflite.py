import argparse
import os
import tensorflow as tf
from pathlib import Path




def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-model_path", help = "Prefix folder name for output", type=str, default=None)
    parser.add_argument("-tflite_name", help = "Prefix folder name for output", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_path = args.model_path
    tflite_name = args.tflite_name
    path = Path(model_path)
    parent_folder = str(path.parent.absolute())
    model_tflite_name = os.path.join(parent_folder, tflite_name)

    print("Starting TFLite Conversion")
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    # Save the TF Lite model.
    with tf.io.gfile.GFile(model_tflite_name, 'wb') as f:
        f.write(tflite_model)
    
    print("TFLite Conversion Done")
    print("Tflite model saved at: ", model_tflite_name)

if __name__ == '__main__':
    main()