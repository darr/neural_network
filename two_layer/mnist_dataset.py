#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : mnist_dataset.py
# Create date : 2018-12-24 19:58
# Modified date : 2019-01-03 10:45
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

#http://yann.lecun.com/exdb/mnist/

#import sys
import struct
import numpy as np
import base_set

# pylint: disable=bad-continuation
meta_lt = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ]
# pylint: enable=bad-continuation


def _get_file_header_data(file_obj, header_len, unpack_str):
    raw_header = file_obj.read(header_len)
    header_data = struct.unpack(unpack_str, raw_header)
    return header_data

class MnistSet(base_set.BaseSet):
    def __init__(self, dic):
        super(MnistSet, self).__init__(dic)
        # pylint: disable=bad-continuation
        self._train_file_list = [
                                "train-images-idx3-ubyte",
                                "train-labels-idx1-ubyte",
                                ]
        self._test_file_list = [
                                "t10k-images-idx3-ubyte",
                                "t10k-labels-idx1-ubyte",
                                ]
        # pylint: enable=bad-continuation

    def _generate_a_batch(self, images_file_name, labels_file_name, batch_size):
        images_file = self._read_file(images_file_name)
        header_data = _get_file_header_data(images_file, 16, ">4I")
        labels_file = self._read_file(labels_file_name)
        header_data = _get_file_header_data(labels_file, 8, ">2I")

        ret = True
        while True:
            images = np.zeros(shape=(batch_size, self.img_size))
            labels = np.zeros(shape=(batch_size, self.class_nums))
            for i in range(batch_size):
                try:
                    image = base_set._read_a_image(images_file, self.img_size)
                    label = base_set._read_a_label(labels_file, self.label_size)
                    images[i] = image
                    labels[i][label] = 1
                except Exception as err:
                    #print(err)
                    ret = False
                    break
            yield images, labels.astype(int), ret

    def _generator_images(self, images_file_name, labels_file_name, path):
        images_file = self._read_file(images_file_name)
        header_data = _get_file_header_data(images_file, 16, ">4I")
        labels_file = self._read_file(labels_file_name)
        header_data = _get_file_header_data(labels_file, 8, ">2I")

        ret = True
        count = 0
        while True:
            try:
                image = base_set._read_a_image(images_file, self.img_size)
                label = base_set._read_a_label(labels_file, self.label_size)
                full_path_name = base_set._get_image_full_name(path, label, count, meta_lt)
                image = np.array(image)
                image = image.reshape(28, 28)
                image = image/ 255.
                base_set.save_image(image, full_path_name)
            except Exception as err:
                print(err)
                ret = False
                break
            count += 1


    def generator_train_images(self, path):
        images_file_name = self._train_file_list[0]
        labels_file_name = self._train_file_list[1]
        self._generator_images(images_file_name, labels_file_name, path)

    def generator_test_images(self, path):
        images_file_name = self._test_file_list[0]
        labels_file_name = self._test_file_list[1]
        self._generator_images(images_file_name, labels_file_name, path)

    def get_train_data_generator(self, batch_size=128):
        images_file_name = self._train_file_list[0]
        labels_file_name = self._train_file_list[1]
        gennerator = self._generate_a_batch(images_file_name, labels_file_name, batch_size)
        return gennerator

    def get_test_data_generator(self, batch_size=128):
        images_file_name = self._test_file_list[0]
        labels_file_name = self._test_file_list[1]
        gennerator = self._generate_a_batch(images_file_name, labels_file_name, batch_size)
        return gennerator
