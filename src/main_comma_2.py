import os
import random
import pandas as pd
import cv2
import idx2numpy
import numpy as np
from tqdm import tqdm
from random import randrange


test_path = ['../data/t10k-images-idx3-ubyte', '../data/t10k-labels-idx1-ubyte']
train_path = ['../data/train-images-idx3-ubyte', '../data/train-labels-idx1-ubyte']
output_dir = '../output_comma_2'
label_file = 'labels.csv'

comma_train_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/experimental/comma_train.csv"
comma_test_path = "/Users/aashishkumar/Documents/Projects/forked_repos/multi-mnist/experimental/comma_test.csv"

os.makedirs(output_dir, exist_ok=True)
n_samples_train = [0,  100,  200,  300, 40000//2, 50000//2, 60000//2, 70000//2, 80000//2]
n_samples_test  = [0,  100,  200,  300,  4000//2,  5000//2,  6000//2,  7000//2,  8000//2]

cnt = 0
number_of_samples_per_class = 10
overlap_size = 30

scale_map = {1 : 0.7, 2 : 0.8, 3 : 0.9, 4 : 1}

def remove_zero_padding(arr, y_start_upper=True, scale=1, comma_present=False):
    """
    Remove all zero padding in the left and right bounding of arr
    :param arr: image as numpy array
    :return: image as numpy array
    """

    left_bounding = 0
    right_bounding = 0

    t = 0
    for j in range(arr.shape[1]):
        if t == 1:
            break

        for i in range(arr.shape[0]):
            if not arr[i][j] == 0:
                left_bounding = j
                t = 1
                break

    t = 0
    for j in reversed(range(arr.shape[1])):
        if t == 1:
            break

        for i in range(arr.shape[0]):
            if not arr[i][j] == 0:
                right_bounding = j
                t = 1
                break
    
    top_bounding = 0
    bottom_bounding = 0
    if comma_present:
        t = 0
        for i in range(arr.shape[0]):
            if t == 1:
                break
            for j in range(arr.shape[1]):
                if not arr[i][j] == 0:
                    top_bounding = i
                    t = 1
                    break
        
        t = 0
        for i in reversed(range(arr.shape[0])):
            if t == 1:
                break
            for j in range(arr.shape[1]):
                if not arr[i][j] == 0:
                    bottom_bounding = i
                    t = 1
                    break
    
    left_bounding = max(0, left_bounding - randrange(0,3))
    right_bounding = min(right_bounding + randrange(0,3), arr.shape[1])

    if comma_present:
        top_bounding = max(0, top_bounding - randrange(0,3))
        bottom_bounding = min(bottom_bounding + randrange(0,3), arr.shape[1])
        temp_arr = np.zeros((28,right_bounding - left_bounding), dtype='uint8')
        height_of_comma = bottom_bounding - top_bounding
        start_of_comma_y = 28 - height_of_comma
        temp_arr[start_of_comma_y:, :] = arr[top_bounding:bottom_bounding, left_bounding:right_bounding]
    else:
        temp_arr = arr[:, left_bounding:right_bounding]
    new_shape_x = max(1,int(temp_arr.shape[1]*scale))
    new_shape_y = max(1,int(temp_arr.shape[0]*scale))
    if new_shape_x > 0 and new_shape_y > 0:
        temp_arr2 = cv2.resize(temp_arr, (new_shape_x, new_shape_y))
    else:
        temp_arr2 = temp_arr

    im1 = np.zeros((28, temp_arr2.shape[1]))
    diff_height = im1.shape[0] - temp_arr2.shape[0]
    # start_y = randrange(diff_height+1)
    start_y = 0
    if y_start_upper == True:
        start_y = 0
    else:
        start_y = diff_height
    
    im1[start_y : start_y +  temp_arr2.shape[0], :] = temp_arr2
    return im1




def print_arr(arr):
    """
    Print out numpy array
    :param arr: numpy array
    :return: void
    """

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print(arr[i][j], end='')
        print()


def concat(a, b, overlap=True, intersection_scalar=0.2):
    """
    Concatenate 2 numpy array
    :param a: numpy array
    :param b: numpy array
    :param overlap: decide 2 array are overlap or not
    :param intersection_scalar: percentage of overlap size
    :return: numpy array
    """

    assert a.shape[0] == b.shape[0]

    if overlap is False:
        return np.concatenate((a, b), axis=1)

    sequence_length = a.shape[1] + b.shape[1]
    intersection_size = int(intersection_scalar * min(a.shape[1], b.shape[1]))

    im = np.zeros((a.shape[0], sequence_length - intersection_size))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not a[i][j] == 0:
                im[i][j] = a[i][j]

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if not b[i][j] == 0:
                im[i][j + (a.shape[1] - intersection_size)] = b[i][j]

    return im


def merge(list_file, overlap_prob=True, y_start_upper=True, scale=1, comma_positions=[]):
    """
    Merge all images in list_file into 1 file
    :param list_file: list of images as numpy array
    :param overlap_prob: decide merged images is overlap or not
    :return: void
    """

    im = np.zeros((28, 1))

    for (i, arr) in enumerate(list_file):
        comma_present = False
        if i in comma_positions:
            comma_present = True
        arr = remove_zero_padding(arr, y_start_upper, scale, comma_present)

        ins = 0
        ovp = False

        overlap = False
        if (i in comma_positions) or (i - 1 in comma_positions):
            overlap = True

        if overlap is True:
            t = random.randint(1, overlap_size)
            ins = float(t / 100)

        if overlap is True:
            ovp = random.choice([True, False])

        im = concat(im, arr, ovp, intersection_scalar=ins)

    return im


def generator(images, labels, n_digits, n_samples, file_out, comma_images, name='train', overlap=True):
    """
    Generate a bunch of data set
    :param images: list of all images
    :param labels: list of labels
    :param n_digits: number of digit will be combined
    :param n_samples: number of samples will be generated
    :param file_out: file label output stream
    :param name: name of directory contain all samples
    :param overlap: decide this samples overlap or not
    :return: void
    """

    global cnt
    save_dir = os.path.join(output_dir, name, str(n_digits))
    os.makedirs(save_dir, exist_ok=True)

    idx = [x for x in range(len(images))]
    comma_idx = [x for x in range(comma_images.shape[0])] 

    for i in tqdm(range(n_samples), desc='Generate ' + str(n_samples) + ' samples'):
        samples = random.choices(idx, k=n_digits)
        
        dig_list = [images[x] for x in samples]
        lab_list = [labels[x] for x in samples]
        comma_positions = []
        if n_digits == 4:
            comma_img_samples = random.choices(comma_idx, k=1)
            dig_list.insert(1, comma_images[comma_img_samples[0]])
            lab_list.insert(1,'a')
            comma_positions = [1]
        elif n_digits == 5:
            comma_img_samples = random.choices(comma_idx, k=1)
            dig_list.insert(2, comma_images[comma_img_samples[0]])
            lab_list.insert(2,'a')
            comma_positions = [2]
        elif n_digits > 5:
            comma_img_samples = random.choices(comma_idx, k=1)
            dig_list.insert(n_digits - 3, comma_images[comma_img_samples[0]])
            lab_list.insert(n_digits - 3,'a')
            # comma_positions = [n_digits - 3]
            
            indian_system = random.choice([True, False])
            comma_gap = 3
            if indian_system:
                comma_gap = 2
            k_idx = 1
            while (n_digits - 3 - (comma_gap * k_idx) > 0):
                new_comma_index = n_digits - 3 - (comma_gap * k_idx)
                comma_img_samples = random.choices(comma_idx, k=1)
                dig_list.insert(new_comma_index, comma_images[comma_img_samples[0]])
                lab_list.insert(new_comma_index,'a')
                # comma_positions.append(new_comma_index)
                k_idx += 1


        comma_positions = [temp_i for temp_i, temp_x in enumerate(lab_list) if temp_x == 'a']
        y_start_upper = False
        if i%2==0:
            y_start_upper = True
        
        scale = scale_map[randrange(1,5)]
        im_mer = merge(dig_list, overlap, y_start_upper, scale, comma_positions)
        im_mer = np.concatenate((np.zeros((28, 2)), im_mer), axis=1)
        im_mer = np.concatenate((im_mer, np.zeros((28, 2))), axis=1)
        im_mer = 255 - im_mer
        cv2.imwrite(os.path.join(save_dir, "comma_" + str(cnt) + '.png'), im_mer)

        lb = ""
        for x in lab_list:
            lb += str(x)

        file_out.write("comma_" + str(cnt) + '.png,' + lb + '\n')
        cnt += 1


if __name__ == '__main__':
    train_file = idx2numpy.convert_from_file(train_path[0])
    train_label = idx2numpy.convert_from_file(train_path[1])
    test_file = idx2numpy.convert_from_file(test_path[0])
    test_label = idx2numpy.convert_from_file(test_path[1])

    comma_train_df = pd.read_csv(comma_train_path, header = None)
    comma_test_df = pd.read_csv(comma_test_path, header = None)

    comma_train_samples = comma_train_df.shape[0]
    comma_test_samples = comma_test_df.shape[0]
    
    comma_train_images = np.asarray(comma_train_df.iloc[:comma_train_samples,:]).reshape([comma_train_samples,28,28]) # taking all columns expect column 0
    comma_train_images = 255 - comma_train_images
    comma_train_images = comma_train_images.astype(np.uint8)

    comma_test_images = np.asarray(comma_test_df.iloc[:comma_test_samples,:]).reshape([comma_test_samples,28,28]) # taking all columns expect column 0
    comma_test_images = 255 - comma_test_images
    comma_test_images = comma_test_images.astype(np.uint8)


    print('------------------Read data----------------------')
    print('Train shape:', train_file.shape)
    print('Train label shape:', train_label.shape)
    print('Test shape:', test_file.shape)
    print('Test label shape:', test_label.shape)
    print('-------------------------------------------------\n')

    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    print('--------------Generate train set-----------------')
    writer = open(os.path.join(output_dir, 'train', label_file), 'w+')

    for num_digits in range(4, 9):
        generator(train_file, train_label, num_digits, n_samples_train[num_digits], writer, comma_train_images, name='train', overlap=False)

    writer.close()
    print('-------------------------------------------------\n')

    print('--------------Generate test set------------------')
    writer = open(os.path.join(output_dir, 'test', label_file), 'w+')

    for num_digits in range(4, 9):
        generator(test_file, test_label, num_digits, n_samples_test[num_digits], writer, comma_test_images, name='test', overlap=False)

    writer.close()
    print('-------------------------------------------------\n')
