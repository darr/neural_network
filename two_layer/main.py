#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2018-12-23 16:53
# Modified date : 2019-01-04 17:14
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import base_set
import cifar10_dataset
import mnist_dataset
import forward_neural_network

def get_cifar10_dic():
    dic = {}
    dic["file_path"] = "./cifar10_data/"
    dic["label_size"] = 1
    dic["class_nums"] = 10
    dic["img_size"] = 32*32*3
    return dic

def get_mnist_dic():
    dic = {}
    dic["file_path"] = "./mnist_data/"
    dic["label_size"] = 1
    dic["class_nums"] = 10
    dic["img_size"] = 28*28
    return dic

def test_cifar10_train_set():
    dic = get_cifar10_dic()
    batch_size = 100
    dataset = cifar10_dataset.Cifar10Set(dic)
    data_generator = dataset.get_train_data_generator(batch_size)
    count = 1
    while True:
        batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
        print("count:%s status:%s " % (count, status))
        if not status:
            break
        count += 1
        #print(str(batch_labels))

def test_cifar10_test_set():
    dic = get_cifar10_dic()
    batch_size = 100
    dataset = cifar10_dataset.Cifar10Set(dic)
    data_generator = dataset.get_test_data_generator(batch_size)
    count = 1
    while True:
        batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
        print("count:%s status:%s " % (count, status))
        if not status:
            break
        count += 1
        #print(str(batch_labels))

def test_generator_mnist_images():
    test_generator_mnist_train_images()
    test_generator_mnist_test_images()

def test_generator_mnist_train_images():
    dic = get_mnist_dic()
    train_img_path = "./mnist_img/train/"
    dataset = mnist_dataset.MnistSet(dic)
    dataset.generator_train_images(train_img_path)

def test_generator_mnist_test_images():
    dic = get_mnist_dic()
    test_img_path = "./mnist_img/test/"
    dataset = mnist_dataset.MnistSet(dic)
    dataset.generator_test_images(test_img_path)

def test_generator_cifar10_images():
    test_generator_cifar10_train_images()
    test_generator_cifar10_test_images()

def test_generator_cifar10_train_images():
    dic = get_cifar10_dic()
    train_img_path = "./cifar10_img/train/"
    dataset = cifar10_dataset.Cifar10Set(dic)
    dataset.generator_train_images(train_img_path)

def test_generator_cifar10_test_images():
    dic = get_cifar10_dic()
    test_img_path = "./cifar10_img/test/"
    dataset = cifar10_dataset.Cifar10Set(dic)
    dataset.generator_test_images(test_img_path)

def test_deep_model_with_epochs():
    neural_model = forward_neural_network.DeepModel()
    neural_model.show_hyperparameters()
    neural_model.run_with_epoch()

def test_deep_model_with_steps(dic):
    neural_model = forward_neural_network.DeepModel(dic)
    neural_model.show_hyperparameters()
    neural_model.run_with_steps()

def test_learn_rate(dic):
#   for i in range(1,6):
#       dic["learn_rate"] = 0.001*i
#       test_deep_model_with_steps(dic)

#   for i in range(1,9):
#       dic["learn_rate"] = 0.0001*i
#       test_deep_model_with_steps(dic)
    dic["learn_rate"] = 0.001
    test_deep_model_with_steps(dic)


def test_activation(dic):
    dic["activation"] = "sigmoid"
    test_learn_rate(dic)
    dic["activation"] = "relu"
    test_learn_rate(dic)

def test_hidden_layer(dic):
    for i in range(6,12):
        hidden_layer = [2**i]
        dic["hidden_layer"] = hidden_layer
        test_activation(dic)

def test_batch_size(dic):
    for i in range(6,10):
        dic["batch_size"] = 2**i
        test_hidden_layer(dic)

def test_mnist():
    dic = {}
    dic["dataset"] = "cifar10"
    test_batch_size(dic)

def test_cifar10():
    dic = {}
    dic["dataset"] = "cifar10"
    dic["activation"] = "relu"
    dic["hidden_layer"] = [2046]
    dic["batch_size"] = 128
    dic["learn_rate"] = 0.001
    test_deep_model_with_steps(dic)
    dic["batch_size"] = 256
    dic["hidden_layer"] = [4096]

    test_deep_model_with_steps(dic)

def run():
    #test_cifar10_train_set()
    #test_cifar10_test_set()
    #test_generator_cifar10_images()
    #test_generator_mnist_images()

    #test_deep_model_with_epochs()
    #test_mnist()
    test_cifar10()

run()

