#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : cifar10_dataset.py
# Create date : 2018-12-24 19:58
# Modified date : 2018-12-31 16:36
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function
#http://www.cs.toronto.edu/~kriz/cifar.html
import sys
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=bad-continuation
meta_lt = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
        ]
# pylint: enable=bad-continuation
def create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def open_file_with_full_name(full_path, open_type):
    try:
        file_object = open(full_path, open_type)
        return file_object
    except Exception as e:
        print(e)
        return None

def get_file_full_name(path, name):
    if path[-1] == "/":
        full_name = path +  name
    else:
        full_name = path + "/" +  name
    return full_name

def open_file(path, name, open_type='a'):
    file_name = get_file_full_name(path, name)
    return open_file_with_full_name(file_name, open_type)

def _get_file_header_data(file_obj, header_len, unpack_str):
    raw_header = file_obj.read(header_len)
    header_data = struct.unpack(unpack_str, raw_header)
    return header_data

def _read_a_image(file_object):
    raw_img = file_object.read(32 * 32)
    red_img = struct.unpack(">1024B", raw_img)

    raw_img = file_object.read(32 * 32)
    green_img = struct.unpack(">1024B", raw_img)

    raw_img = file_object.read(32 * 32)
    blue_img = struct.unpack(">1024B", raw_img)

    img = np.zeros(shape=(1024, 3))
    for i in range(1024):
        l = [red_img[i], green_img[i], blue_img[i]]
        img[i] = l
    img = img.reshape(32, 32, 3)
    img = img / 255.
    return img

def _read_one_image(file_object):
    raw_img = file_object.read(32 * 32 * 3)
    img = struct.unpack(">3072B", raw_img)
    return img

def _read_a_label(file_object):
    raw_label = file_object.read(1)
    label = struct.unpack(">B", raw_label)
    return label

def _get_image_full_name(path, label, count):
    meta = meta_lt[label[0]]
    full_path = "%s%s" %(path, meta)
    create_path(full_path)
    full_path_name = "%s/%s.jpg" %(full_path, count)
    return full_path_name

def save_image(image, full_path_name):
    plt.imshow(image)
    plt.savefig(full_path_name)
    plt.close()

class Cifar10Set(object):
    def __init__(self, file_path):
        super(Cifar10Set, self).__init__()
        # pylint: disable=bad-continuation
        self._train_file_list = [
                            "data_batch_1.bin",
                            "data_batch_2.bin",
                            "data_batch_3.bin",
                            "data_batch_4.bin",
                            "data_batch_5.bin"
                            ]
        # pylint: enable=bad-continuation
        self._test_file_list = ["test_batch.bin",]
        self.file_path = file_path

    def _read_file(self, file_name):
        file_object = open_file(self.file_path, file_name, open_type="rb")
        return file_object

    def _generate_a_batch(self, batch_size, file_list):
        images = np.zeros(shape=(batch_size, 32 * 32 * 3))
        labels = np.zeros(shape=(batch_size, 10))
        i = 0
        file_name = file_list[i]
        file_name = "cifar-10-batches-bin/%s" % file_name
        train_file = self._read_file(file_name)

        count = 0
        ret = True
        while True:
            while count < batch_size:
                try:
                    label = _read_a_label(train_file)
                    image = _read_one_image(train_file)
                    images[count] = image
                    labels[count][label[0]] = 1
                    count += 1
                except Exception as err:
                    #print(err)
                    if i >= len(self._train_file_list):
                        ret = False
                        break
                    else:
                        i += 1
                        if i < len(file_list):
                            file_name = file_list[i]
                            file_name = "cifar-10-batches-bin/%s" % file_name
                            train_file = self._read_file(file_name)
            count = 0
            yield images, labels.astype(int), ret
            images = np.zeros(shape=(batch_size, 32*32*3))
            labels = np.zeros(shape=(batch_size, 10))

    def generator_images(self, file_list, path):
        count = 1
        for i in range(len(file_list)):
            file_name = file_list[i]
            file_name = "cifar-10-batches-bin/%s" % file_name
            train_file = self._read_file(file_name)

            while True:
                try:
                    label = _read_a_label(train_file)
                    image = _read_a_image(train_file)
                    full_path_name = _get_image_full_name(path, label, count)
                    save_image(image, full_path_name)
                    print("file:%s count:%s"% (file_name, count))

                except Exception as err:
                    print(err)
                    break
                count += 1

    def generator_train_images(self, path):
        self.generator_images(self._train_file_list, path)

    def generator_test_images(self, path):
        self.generator_images(self._test_file_list, path)

    def get_train_data_generator(self, batch_size=128):
        file_list = self._train_file_list
        gennerator = self._generate_a_batch(batch_size, file_list)
        return gennerator

    def get_test_data_generator(self, batch_size=128):
        file_list = self._test_file_list
        gennerator = self._generate_a_batch(batch_size, file_list)
        return gennerator

    def get_a_batch_data(self, data_generator):
        if sys.version > '3':
            batch_img, batch_labels, status = data_generator.__next__()
        else:
            batch_img, batch_labels, status = data_generator.next()
        return batch_img, batch_labels, status
