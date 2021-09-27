import os
import cv2
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
import click

class MultiDigitDataLoaderRealData(tf.keras.utils.Sequence):
    def __init__(self, 
                 df_path, 
                 image_base_path, 
                 transform,
                 shuffle=True):
        '''
        df_path: path for dataframe which has file_name and labels
        image_base_path: folder where images are
        batch_size: batch_size while training
        img_height: height of image
        img_width: width of image
        num_time_steps = number of input time steps for lstm layer
        '''
        self.image_base_path = image_base_path
        self.shuffle = shuffle
        self.transform = transform
        
        self.char_to_int_map = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10}
        
        self.df = pd.read_csv(df_path, header = None, dtype={0: str, 1: str})
        self.df.columns = ["file_name", "labels"]
        self.df['label_length'] = self.df['labels'].apply(self.get_label_length)
        
        ## shuffle the df
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def get_label_length(self, label):
        return len(label)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def get_image(self, filename):
        img = cv2.imread(filename, 0)
        img = self.transform(image=img)["image"]
        img_height = img.shape[0]
        img_width = img.shape[1]
        img = img.reshape((img_height, img_width, 1))
        return img
    
    def __getitem__(self, idx):        
        x_train = [self.get_image(self.image_base_path + self.df['file_name'][idx])]
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, axes=[0,2,1,3])
        
        input_length_arr =  (x_train.shape[1] // 4) * np.ones(shape=(1, 1), dtype="int64")
        
        label_length_arr = self.df['label_length'][idx].reshape(1 ,1)
        
        y_train = np.ones([1,self.df['label_length'][idx]]) * (-1)
        # num_list = [int(char) for char in self.df['labels'][idx]]
        num_list = [self.char_to_int_map[char] for char in self.df['labels'][idx]]
        y_train[0, 0:self.df['label_length'][idx]] = num_list
        
        inputs = {'image': x_train,
                  'label': y_train,
                  'input_length': input_length_arr,
                  'label_length': label_length_arr,
                  }
        
        return inputs


def check_dataset(data):
    for batch in data:
        print("x_train.shape",batch['image'].shape)
        print("y_train.shape",batch['label'].shape)
        print("input_length_arr.shape",batch['input_length'].shape)
        print("label_length_arr.shape",batch['label_length'].shape)
        break

def decode_batch_predictions(pred):
    char_to_int_map = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10}
    inv_map = {v: k for k, v in char_to_int_map.items()}
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    results = results.numpy()
    # Iterate over the results and get back the text
    output_preds = []
    for res in results:
        pred_str = ""
        for i in res:
            if i != -1:
                pred_str += inv_map[i]
                # pred_str += str(i)
        output_preds.append(pred_str)
    return output_preds

def get_preds(dataset, model_test):
    pred_texts = []
    for i, batch in enumerate(dataset):
        if i%1000==0:
            print(i)
        batch_images = batch["image"]
        preds = model_test.predict(batch_images)
        pred_texts_temp = decode_batch_predictions(preds)
        pred_texts += pred_texts_temp
    return pred_texts


def editDistDP(str1, str2): 
    m = len(str1)
    n = len(str2)
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m + 1): 
        for j in range(n + 1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n] 

def get_edit_distance_freq(df):
    edit_distance_freq = {}
    wrong_count = 0
    for i in range(0, df.shape[0]):
        if df['labels'][i] is not None:
            edit_distance = editDistDP(df['labels'][i], df['preds'][i])
            if edit_distance_freq.get(edit_distance) is None:
                edit_distance_freq[edit_distance] = 1
            else:
                edit_distance_freq[edit_distance] += 1
            if edit_distance > 2:
                print(df['file_name'][i],df['labels'][i], df['preds'][i], i)
            if df['labels'][i] != df['preds'][i]:
                wrong_count += 1
    
    return edit_distance_freq, wrong_count
    
def get_word_leng_freq(df):
    correct_word_length_freq = {}
    wrong_word_length_freq = {}
    for i in range(0, df.shape[0]):
        if df['labels'][i] == df['preds'][i]:
            word_length = len(df['labels'][i])
            if correct_word_length_freq.get(word_length) is None:
                correct_word_length_freq[word_length] = 1
            else:
                correct_word_length_freq[word_length] += 1
        else:
            word_length = len(df['labels'][i])
            if wrong_word_length_freq.get(word_length) is None:
                wrong_word_length_freq[word_length] = 1
            else:
                wrong_word_length_freq[word_length] += 1
        
    return correct_word_length_freq, wrong_word_length_freq

def get_loss_plot(history_1, figure_folder=None):
    plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
    plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set
    plt.title("Loss Curve",fontsize=18)
    plt.xlabel("Epochs",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(figure_folder, "loss.png"))

def create_loss_csv(history_1, figure_folder=None):
    loss_df = pd.DataFrame(list(zip(history_1.epoch, history_1.history['loss'], history_1.history['val_loss'])), 
                            columns = ['Epoch', 'Train_Loss', 'Val_loss'])
    loss_df.to_csv(os.path.join(figure_folder, "loss.csv"), index=None)

def show_edit_distance_freq_graph(edit_distance_freq, title, figure_folder=None):
    plt.figure(figsize=(7,5))
    plt.bar(edit_distance_freq.keys(),edit_distance_freq.values(),width = 0.8,color="orange")
    for i in edit_distance_freq:
        plt.text(i,edit_distance_freq[i],str(edit_distance_freq[i]),horizontalalignment='center',fontsize=10)

    plt.tick_params(labelsize = 14)
    plt.xticks(range(0,8))
    plt.xlabel("Edit Distance",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.title(title)
    figure_name = title + ".png"
    plt.savefig(os.path.join(figure_folder, figure_name))

def show_word_length_freq_graph(freq_var, title, figure_folder=None):
    plt.figure(figsize=(7,5))
    plt.bar(freq_var.keys(),freq_var.values(),width = 0.8,color="orange")
    for i in freq_var:
        plt.text(i,freq_var[i],str(freq_var[i]),horizontalalignment='center',fontsize=10)

    plt.tick_params(labelsize = 14)
    plt.xticks(range(0,10))
    plt.xlabel("word length",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.title(title)
    figure_name = title + ".png"
    plt.savefig(os.path.join(figure_folder, figure_name))


@click.command()
@click.option('--output_dir', default=None, help="Output directory to store artifacts")
def main(output_dir):
    train_df_path =         "/home/kaushikdas/aashish/multi-digit-dataset/real_data_hw/set-3/train/labels.csv"
    train_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/real_data_hw/set-3/train/all/"

    test_df_path =         "/home/kaushikdas/aashish/multi-digit-dataset/real_data_hw/set-3/test/labels.csv"
    test_image_base_path = "/home/kaushikdas/aashish/multi-digit-dataset/real_data_hw/set-3/test/all/"

    shuffle = True

    train_transform = A.Compose([
                                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.1, p=1),
                                A.ShiftScaleRotate(shift_limit_x=0.01, shift_limit_y=0.1, rotate_limit=5),
                                A.ToFloat(max_value=255)
                                ])
    
    test_transform = A.Compose([
                                A.ToFloat(max_value=255)
                              ])
    
    train_multi_digit_dataset = MultiDigitDataLoaderRealData(df_path = train_df_path,
                                                             image_base_path = train_image_base_path,
                                                             transform = train_transform,
                                                             shuffle = shuffle) 

    test_multi_digit_dataset = MultiDigitDataLoaderRealData(df_path = test_df_path,
                                                            image_base_path = test_image_base_path,
                                                            transform = test_transform,
                                                            shuffle = shuffle)
    
    check_dataset(train_multi_digit_dataset)
    check_dataset(test_multi_digit_dataset)

    pretrained_model_path = "/home/kaushikdas/aashish/multi-mnist/saved_models/transformer_try1/multi_digit_model_1_to_8_comma_transformer_original"
    model = tf.keras.models.load_model(pretrained_model_path)
    print(model.summary())

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    stochastic_avg_opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer=stochastic_avg_opt)

    epochs = 10
    lrr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,verbose=1,factor=0.5, min_lr=0.000001)
    # Train the model
    history_1 = model.fit(
                          train_multi_digit_dataset, 
                          validation_data=test_multi_digit_dataset, 
                          epochs=epochs, 
                          callbacks=[lrr]
                          )
    
    get_loss_plot(history_1, output_dir)
    create_loss_csv(history_1, output_dir)

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    print(prediction_model.summary())

    train_pred_texts = get_preds(train_multi_digit_dataset, prediction_model)
    df_train_labels = train_multi_digit_dataset.df.copy()
    df_train_labels['preds'] = train_pred_texts

    test_pred_texts = get_preds(test_multi_digit_dataset, prediction_model)
    df_test_labels = test_multi_digit_dataset.df.copy()
    df_test_labels['preds'] = test_pred_texts
    
    print("\nCalculating Edit Distance\n")
    train_edit_distance_freq, train_wrong_count = get_edit_distance_freq(df_train_labels)
    show_edit_distance_freq_graph(train_edit_distance_freq, "train_edit_distance_frequency", output_dir)
    test_edit_distance_freq, test_wrong_count = get_edit_distance_freq(df_test_labels)
    show_edit_distance_freq_graph(test_edit_distance_freq, "test_edit_distance_frequency", output_dir)

    print("\nCalculating Word Length Frequency\n")
    train_correct_word_length_freq, train_wrong_word_length_freq = get_word_leng_freq(df_train_labels)
    show_word_length_freq_graph(train_wrong_word_length_freq,"train_wrong_word_length_frequency", output_dir)
    show_word_length_freq_graph(train_correct_word_length_freq,"train_correct_word_length_frequency", output_dir)

    test_correct_word_length_freq, test_wrong_word_length_freq = get_word_leng_freq(df_test_labels)
    show_word_length_freq_graph(test_wrong_word_length_freq,"test_wrong_word_length_frequency", output_dir)
    show_word_length_freq_graph(test_correct_word_length_freq,"test_correct_word_length_frequency", output_dir)

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

    model_folder_name = os.path.join(output_dir, "multi_digit_model_1_to_8_comma_transformer")
    model_json_file_name = os.path.join(output_dir, "multi_digit_model_1_to_8_comma_transformer_json.json")
    model_weights_file_name =  os.path.join(output_dir, "multi_digit_model_1_to_8_comma_transformer_weights.h5")
    # model_tflite_name = os.path.join(output_dir, "worksheet.multi_digit_ocr.tflite")
    model_tflite_name = os.path.join(output_dir, "worksheet.multi_digit_model_1_to_8_comma_transformer_retrain.tflite")

    print("\n\nSaving Original Model\n\n")
    model.save(model_folder_name + "_original")
        
    print("\n\nSaving Prediction Model\n\n")
    prediction_model.save(model_folder_name)

    # model_json = prediction_model.to_json()
    # with open(model_json_file_name, "w") as json_file:
    #     json_file.write(model_json)

    prediction_model.save_weights(model_weights_file_name)

    # convert to tflite model
    print("\n\nConverting to TF Lite \n\n")
    converter = tf.lite.TFLiteConverter.from_keras_model(prediction_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with tf.io.gfile.GFile(model_tflite_name, 'wb') as f:
        f.write(tflite_model)



if __name__ == '__main__':
    main()
