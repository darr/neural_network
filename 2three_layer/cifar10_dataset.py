#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : cifar10_dataset.py
# Create date : 2018-12-24 19:58
# Modified date : 2019-01-03 12:04
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function
#http://www.cs.toronto.edu/~kriz/cifar.html

import numpy as np
import base_set

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

def _read_the_image(image):
    red_img = image[0 : 32*32]
    green_img = image[32*32 : 32*32*2]
    blue_img = image[32*32*2 : 32*32*3]

    img = np.zeros(shape=(32*32, 3))
    for i in range(1024):
        l = [red_img[i], green_img[i], blue_img[i]]
        img[i] = l
    img = img.reshape(32, 32, 3)
    img = img / 255.
    return img

class Cifar10Set(base_set.BaseSet):
    def __init__(self, dic):
        super(Cifar10Set, self).__init__(dic)
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

    def _generate_a_batch(self, batch_size, file_list):
        images = np.zeros(shape=(batch_size, self.img_size))
        labels = np.zeros(shape=(batch_size, self.class_nums))
        i = 0
        file_name = file_list[i]
        file_name = "cifar-10-batches-bin/%s" % file_name
        train_file = self._read_file(file_name)

        count = 0
        ret = True
        while True:
            while count < batch_size:
                try:
                    label = base_set._read_a_label(train_file, self.img_size)
                    image = base_set._read_a_image(train_file, self.label_size)
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

    def _generator_images(self, file_list, path):
        count = 1
        for i in range(len(file_list)):
            file_name = file_list[i]
            file_name = "cifar-10-batches-bin/%s" % file_name
            train_file = self._read_file(file_name)

            while True:
                try:
                    label = base_set._read_a_label(train_file, self.label_size)
                    image = base_set._read_a_image(train_file, self.img_size)
                    image = _read_the_image(image)
                    full_path_name = base_set._get_image_full_name(path, label, count, meta_lt)
                    base_set.save_image(image, full_path_name)
                    print("file:%s count:%s"% (file_name, count))

                except Exception as err:
                    print(err)
                    break
                count += 1

    def generator_train_images(self, path):
        self._generator_images(self._train_file_list, path)

    def generator_test_images(self, path):
        self._generator_images(self._test_file_list, path)

    def get_train_data_generator(self, batch_size=128):
        file_list = self._train_file_list
        gennerator = self._generate_a_batch(batch_size, file_list)
        return gennerator

    def get_test_data_generator(self, batch_size=128):
        file_list = self._test_file_list
        gennerator = self._generate_a_batch(batch_size, file_list)
        return gennerator
