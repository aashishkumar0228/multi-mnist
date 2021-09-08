import os
import random

import cv2
import idx2numpy
import numpy as np
from tqdm import tqdm
from random import randrange


test_path = ['../data/t10k-images-idx3-ubyte', '../data/t10k-labels-idx1-ubyte']
train_path = ['../data/train-images-idx3-ubyte', '../data/train-labels-idx1-ubyte']
output_dir = '../output_4'
label_file = 'labels.csv'

os.makedirs(output_dir, exist_ok=True)
n_samples_train = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
n_samples_test  = [0,  1000,  2000,  3000,  4000,  5000,  6000,  7000,  8000]

cnt = 0
number_of_samples_per_class = 10
overlap_size = 30

scale_map = {1 : 0.7, 2 : 0.8, 3 : 0.9, 4 : 1}

def remove_zero_padding(arr, y_start_upper=True, scale=1):
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
    
    left_bounding = max(0, left_bounding - randrange(0,3))
    right_bounding = min(right_bounding + randrange(0,3), arr.shape[1])

    temp_arr = arr[:, left_bounding:right_bounding]
    new_shape_x = max(1,int(temp_arr.shape[1]*scale))
    new_shape_y = max(1,int(temp_arr.shape[0]*scale))
    temp_arr2 = cv2.resize(temp_arr, (new_shape_x, new_shape_y))

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


def merge(list_file, overlap_prob=True, y_start_upper=True, scale=1):
    """
    Merge all images in list_file into 1 file
    :param list_file: list of images as numpy array
    :param overlap_prob: decide merged images is overlap or not
    :return: void
    """

    im = np.zeros((28, 1))

    for (i, arr) in enumerate(list_file):
        arr = remove_zero_padding(arr, y_start_upper, scale)

        ins = 0
        ovp = False

        if overlap_prob is True:
            t = random.randint(1, overlap_size)
            ins = float(t / 100)

        if overlap_prob is True:
            ovp = random.choice([True, False])

        im = concat(im, arr, ovp, intersection_scalar=ins)

    return im


def generator(images, labels, n_digits, n_samples, file_out, name='train', overlap=True):
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

    for i in tqdm(range(n_samples), desc='Generate ' + str(n_samples) + ' samples'):
        samples = random.choices(idx, k=n_digits)
        dig_list = [images[x] for x in samples]
        lab_list = [labels[x] for x in samples]
        y_start_upper = False
        if i%2==0:
            y_start_upper = True
        
        scale = scale_map[randrange(1,5)]
        im_mer = merge(dig_list, overlap, y_start_upper, scale)
        im_mer = np.concatenate((np.zeros((28, 2)), im_mer), axis=1)
        im_mer = np.concatenate((im_mer, np.zeros((28, 2))), axis=1)
        im_mer = 255 - im_mer
        cv2.imwrite(os.path.join(save_dir, str(cnt) + '.png'), im_mer)

        lb = ""
        for x in lab_list:
            lb += str(x)

        file_out.write(str(cnt) + '.png,' + lb + '\n')
        cnt += 1


if __name__ == '__main__':
    train_file = idx2numpy.convert_from_file(train_path[0])
    train_label = idx2numpy.convert_from_file(train_path[1])
    test_file = idx2numpy.convert_from_file(test_path[0])
    test_label = idx2numpy.convert_from_file(test_path[1])

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

    for num_digits in range(1, 9):
        generator(train_file, train_label, num_digits, n_samples_train[num_digits], writer, name='train', overlap=False)

    writer.close()
    print('-------------------------------------------------\n')

    print('--------------Generate test set------------------')
    writer = open(os.path.join(output_dir, 'test', label_file), 'w+')

    for num_digits in range(1, 9):
        generator(test_file, test_label, num_digits, n_samples_test[num_digits], writer, name='test', overlap=False)

    writer.close()
    print('-------------------------------------------------\n')
