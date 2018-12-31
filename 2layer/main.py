#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2018-12-23 16:53
# Modified date : 2018-12-31 15:37
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import cifar10_dataset
import forward_neural_network

def test_cifar10_train_set():
    file_path = "./data/"
    batch_size = 100
    dataset = cifar10_dataset.Cifar10Set(file_path)
    data_generator = dataset.get_train_data_generator(batch_size)
    count = 1
    while True:
        batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
        print("count:%s status:%s " % (count, status))
        if not status:
            break
        count += 1
        print(str(batch_labels))

def test_cifar10_test_set():
    file_path = "./data/"
    batch_size = 100
    dataset = cifar10_dataset.Cifar10Set(file_path)
    data_generator = dataset.get_test_data_generator(batch_size)
    count = 1
    while True:
        batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
        print("count:%s status:%s " % (count, status))
        if not status:
            break
        count += 1
        print(str(batch_labels))

def test_generator_images():
#    test_generator_train_images()
    test_generator_test_images()

def test_generator_train_images():
    file_path = "./data/"
    train_img_path = "./img/train/"
    dataset = cifar10_dataset.Cifar10Set(file_path)
    dataset.generator_train_images(train_img_path)

def test_generator_test_images():
    file_path = "./data/"
    test_img_path = "./img/test/"
    dataset = cifar10_dataset.Cifar10Set(file_path)
    dataset.generator_test_images(test_img_path)

def test_deep_model_with_epochs():
    neural_model = forward_neural_network.DeepModel()
    neural_model.show_hyperparameters()
    neural_model.run_with_epoch()

def test_deep_model_with_steps():
    neural_model = forward_neural_network.DeepModel()
    neural_model.show_hyperparameters()
    neural_model.run_with_steps()

def run():
#    test_cifar10_train_set()
    #test_cifar10_test_set()
    test_generator_images()
    #test_deep_model_with_epochs()
    #test_deep_model_with_steps()

run()

