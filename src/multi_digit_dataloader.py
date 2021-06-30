import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

class MultiDigitDataLoader(tf.keras.utils.Sequence):
    def __init__(self, 
                 df_path, 
                 image_base_path, 
                 batch_size, 
                 img_height,
                 img_width, 
                 num_time_steps,
                 transform,
                 max_digit_length=3, 
                 shuffle=True):
        '''
        df_path: path for dataframe which has file_name and labels
        image_base_path: folder where images are
        batch_size: batch_size while training
        img_height: height of image
        img_width: width of image
        num_time_steps = number of input time steps for lstm layer
        '''
        self.batch_size = int(batch_size)
        self.image_base_path = image_base_path
        self.shuffle = shuffle
        self.max_digit_length = max_digit_length
        self.img_height = img_height
        self.img_width = img_width
        self.num_time_steps = num_time_steps
        self.transform = transform
        
        self.df = pd.read_csv(df_path, header = None, dtype={0: str, 1: str})
        self.df.columns = ["file_name", "labels"]
        self.df['y_true'] = self.df['labels'].apply(self.create_y_true)
        self.df['label_length'] = self.df['labels'].apply(self.get_label_length)
        
        ## shuffle the df
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def create_y_true(self, label):
        y_true = np.ones([1,self.max_digit_length]) * (-1)
        num_list = [int(char) for char in label]
        y_true[0, 0:len(label)] = num_list
        return y_true
    
    def get_label_length(self, label):
        return len(label)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))
    
    def get_image(self, filename):
        img = cv2.imread(filename, 0)
        img = self.transform(image=img)["image"]
        img = img.reshape((self.img_height, self.img_width, 1))
        return img
    
    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = min((idx + 1) * self.batch_size, len(self.df))
        
        y_train = np.concatenate(self.df['y_true'][idx_start:idx_end].values,axis=0)
        
        x_train = [self.get_image(self.image_base_path + self.df['file_name'][i]) for i in range(idx_start,idx_end)]
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, axes=[0,2,1,3])
        x_train = x_train / 255
        
        input_length_arr = self.num_time_steps * np.ones(shape=(idx_end - idx_start, 1), dtype="int64")
        
        label_length_arr = self.df['label_length'][idx_start:idx_end].values.reshape(idx_end - idx_start,1)
        
        inputs = {'image': x_train,
                  'label': y_train,
                  'input_length': input_length_arr,
                  'label_length': label_length_arr,
                  }
        
        return inputs