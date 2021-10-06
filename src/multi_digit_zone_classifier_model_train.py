import os
import cv2
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A
import tensorflow_addons as tfa
from tqdm import tqdm
import click

# set random seed
seed = 3215
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

class MultiDigitZoneDataLoader(tf.keras.utils.Sequence):
    def __init__(self, 
                 folder_path,
                 num_classes,
                 transform,
                 shuffle=True):
        '''
        folder_path: folder where images are
        num_classes: number of classes
        transform: transformations to apply on image
        shuffle: if true shuffle dataset after every epoch
        '''
        self.folder_path = folder_path
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.transform = transform
        
        img_paths = []
        labels = []
        for i in range(self.num_classes):
            temp_img_paths = glob.glob(os.path.join(self.folder_path, str(i)) + "/*.bmp")
            temp_labels = [i]*len(temp_img_paths)
            img_paths = img_paths + temp_img_paths
            labels = labels + temp_labels

        self.df = pd.DataFrame({'image_path':img_paths,'label':labels})
        
        ## shuffle the df
        if self.shuffle:
            self.shuffle_data()
            
    def shuffle_data(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()
    
    def __len__(self):
        return int(len(self.df))
    
    def get_image(self, filename):
        img = cv2.imread(filename, 0)
        img = self.transform(image=img)["image"]
        img_height = img.shape[0]
        img_width = img.shape[1]
        img = img.reshape((img_height, img_width, 1))
        return img
    
    def __getitem__(self, idx):
        idx_start = idx
        idx_end = min((idx + 1), len(self.df))
                
        y_train = self.df['label'][idx_start:idx_end].values.reshape(idx_end - idx_start,1)
        
        
        x_train = [self.get_image(self.df['image_path'][i]) for i in range(idx_start,idx_end)]
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, axes=[0,2,1,3])
        
        return x_train, keras.utils.to_categorical(y_train, num_classes=self.num_classes)


def build_small_lstm_model(img_height, num_classes):
    # Inputs to the model
    input_img = layers.Input(shape=(None, img_height, 1), name="image", dtype="float32")

    # First conv block
    x = layers.Conv2D(16,(3, 3),padding="same",name="Conv1")(input_img)
    x = layers.ELU(name="elu1")(x)
    x = layers.Conv2D(16,(3, 3),padding="same",name="Conv2")(x)
    x = layers.ELU(name="elu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, seed=142, name="dropout1")(x)

    # Second conv block
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv3")(x)
    x = layers.ELU(name="elu3")(x)
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv4")(x)
    x = layers.ELU(name="elu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.2, seed=148, name="dropout2")(x)
    
    new_shape = (-1, x.shape[2]*x.shape[3])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, seed=196, name="dropout3")(x)
    
    x = layers.Bidirectional(layers.LSTM(32), name="bidirectional1")(x)
    
    output = layers.Dense(num_classes, activation="softmax", name="dense2")(x)
    
    model = keras.Model(inputs=input_img, outputs=output, name="zone_model")
    return model


def get_accuracy(dataset, model):
    correct = 0
    wrong = 0
    for batch in tqdm(dataset):
        x, y = batch
        y_pred = np.argmax(model.predict(x))
        y_true = np.argmax(y)
        if y_true == y_pred:
            correct += 1
        else:
            wrong += 1
    print("Correct", correct, "Wrong", wrong)
    print("accuracy: ", correct/(correct + wrong))
    return correct, wrong

def get_loss_accuracy_plot(history_1, figure_folder=None):
    f = plt.figure(figsize=(20,7))
    #Adding Subplot 1 (For Accuracy)
    f.add_subplot(121)

    plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set
    plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

    plt.title("Accuracy Curve",fontsize=18)
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()

    #Adding Subplot 1 (For Loss)
    f.add_subplot(122)

    plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
    plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set

    plt.title("Loss Curve",fontsize=18)
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(figure_folder, "accuracy_loss.png"))

@click.command()
@click.option('--output_dir', default=None, help="Output directory to store artifacts")
def main(output_dir):
    train_folder_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/zone_classifier_data/set_6/train"
    test_folder_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/zone_classifier_data/set_6/test"
    num_classes = 3
    img_height = 28
    shuffle = True
    epochs = 15
    
    # batch size is 1 as each image width is different

    train_transform = A.Compose([
                                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.1, p=1),
                                A.ShiftScaleRotate(shift_limit_x=0.01, shift_limit_y=0.1, rotate_limit=5),
                                A.ToFloat(max_value=255)
                                ])
    
    train_multi_digit_dataset = MultiDigitZoneDataLoader(folder_path = train_folder_path,
                                                         num_classes = num_classes,
                                                         transform = train_transform,
                                                         shuffle = shuffle)

    test_transform = A.Compose([
                                A.ToFloat(max_value=255)
                                ])
    
    test_multi_digit_dataset = MultiDigitZoneDataLoader(folder_path = test_folder_path,
                                                         num_classes = num_classes,
                                                         transform = test_transform,
                                                         shuffle = shuffle)


    print("Length train_multi_digit_dataset: ",len(train_multi_digit_dataset))
    print("Length test_multi_digit_dataset: ",len(test_multi_digit_dataset))

    for batch in train_multi_digit_dataset:
        x, y = batch
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        break
    
    model = build_small_lstm_model(img_height, num_classes)

    loss = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    moving_avg_opt = tfa.optimizers.MovingAverage(opt)
    model.compile(optimizer=moving_avg_opt, loss=loss ,metrics=['accuracy'])
    print(model.summary())
    print("-"*20)
    print("Training Starts")
    lrr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',patience=2,verbose=1,factor=0.5, min_lr=0.000001)
    history_1 = model.fit(
                          train_multi_digit_dataset, 
                          validation_data=test_multi_digit_dataset, 
                          epochs=epochs, 
                          callbacks=[lrr]
                          )
    print("Training End")
    print("-"*20)
    get_loss_accuracy_plot(history_1, output_dir)

    print("Calculate Accuracy")
    get_accuracy(train_multi_digit_dataset, model)
    print("Moving Average Optimizer Weight Update")
    moving_avg_opt.assign_average_vars(model.variables)
    print("Calculate Accuracy")
    train_correct_count, train_wrong_count = get_accuracy(train_multi_digit_dataset, model)
    test_correct_count, test_wrong_count = get_accuracy(test_multi_digit_dataset, model)
    
    total_train_samples = train_correct_count + train_wrong_count
    total_test_samples = test_correct_count + test_wrong_count
    print("Train wrong:", train_wrong_count)
    print("Train size:", total_train_samples)
    print("Train Error %:",train_wrong_count/total_train_samples*100)
    print("Train Accuracy %:",(1 - (train_wrong_count/total_train_samples))*100)
    print("Test wrong:", test_wrong_count)
    print("Test size:", total_test_samples)
    print("Test Error %:",test_wrong_count/total_test_samples*100)
    print("Test Accuracy %:",(1 - (test_wrong_count/total_test_samples))*100)

    train_accuracy_file = open(os.path.join(output_dir, "train_test_accuracy.txt"), 'w')
    train_accuracy_file.write("Train wrong: {}\n".format(train_wrong_count))
    train_accuracy_file.write("Train size: {}\n".format(total_train_samples))
    train_accuracy_file.write("Train Error %: {}\n".format(train_wrong_count/total_train_samples*100))
    train_accuracy_file.write("Train Accuracy %: {:.2f}\n\n".format((1 - (train_wrong_count/total_train_samples))*100))
    train_accuracy_file.write("Test wrong: {}\n".format(test_wrong_count))
    train_accuracy_file.write("Test size: {}\n".format(total_test_samples))
    train_accuracy_file.write("Test Error %: {}\n".format(test_wrong_count/total_test_samples*100))
    train_accuracy_file.write("Test Accuracy %: {:.2f}\n".format((1 - (test_wrong_count/total_test_samples))*100))
    train_accuracy_file.close()

    model_tflite_name = os.path.join(output_dir, "worksheet.hw_ocr_multi_digits_zone_clf.tflite")
    model_folder_name = os.path.join(output_dir, "multi_digits_zone_classifier")
    model_json_file_name = os.path.join(output_dir, "multi_digits_zone_classifier_json.json")
    model_weights_file_name = os.path.join(output_dir, "multi_digits_zone_classifier_weights.h5")

    print("Saving Model")
    model.save(model_folder_name)
    print("Saving Model Json")
    model_json = model.to_json()
    with open(model_json_file_name, "w") as json_file:
        json_file.write(model_json)
    
    print("Saving Model Weights")
    model.save_weights(model_weights_file_name)

    print("Converting to TFLite")
    converter = tf.lite.TFLiteConverter.from_saved_model(model_folder_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile(model_tflite_name, 'wb') as f:
        f.write(tflite_model)
    
    print("Done")
    

if __name__ == '__main__':
    main()