#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : base_set.py
# Create date : 2018-12-24 19:58
# Modified date : 2019-01-03 11:23
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys
import os
import struct
import matplotlib.pyplot as plt

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

def _read_a_image(file_object, img_size):
    raw_img = file_object.read(img_size)
    img = struct.unpack(">%sB" % img_size, raw_img)
    return img

def _read_a_label(file_object, label_size):
    raw_label = file_object.read(label_size)
    label = struct.unpack(">%sB" % label_size, raw_label)
    return label

def _get_image_full_name(path, label, count, meta_lt):
    meta = meta_lt[label[0]]
    full_path = "%s%s" %(path, meta)
    create_path(full_path)
    full_path_name = "%s/%s.jpg" %(full_path, count)
    print(full_path_name)
    return full_path_name

def save_image(image, full_path_name):
    plt.imshow(image)
    plt.savefig(full_path_name)
    plt.close()

class BaseSet(object):
    def __init__(self, dic):
        super(BaseSet, self).__init__()
        self.file_path = dic["file_path"]
        self.img_size = dic["img_size"]
        self.label_size = dic["label_size"]
        self.class_nums = dic["class_nums"]

    def _read_file(self, file_name):
        file_object = open_file(self.file_path, file_name, open_type="rb")
        return file_object

    def get_a_batch_data(self, data_generator):
        if sys.version > '3':
            batch_img, batch_labels, status = data_generator.__next__()
        else:
            batch_img, batch_labels, status = data_generator.next()
        return batch_img, batch_labels, status
