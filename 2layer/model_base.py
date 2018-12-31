#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : model_base.py
# Create date : 2018-12-25 20:04
# Modified date : 2018-12-31 16:02
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import cifar10_dataset
import func

class ModelBase(object):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.hyper = self._get_hyperparameters()

    def _get_dataset(self):
        dataset = cifar10_dataset.Cifar10Set(self.hyper["file_path"])
        return dataset

    def _get_train_generator_with(self, dataset):
        data_generator = dataset.get_train_data_generator(self.hyper["batch_size"])
        return data_generator

    def _get_test_generator_with(self, dataset):
        data_generator = dataset.get_test_data_generator(self.hyper["batch_size"])
        return data_generator

    def _get_train_generator(self):
        dataset = self._get_dataset()
        data_generator = self._get_train_generator_with(dataset)
        return data_generator

    def _get_test_generator(self):
        dataset = self._get_dataset()
        data_generator = self._get_test_generator_with(dataset)
        return data_generator

    def _get_a_batch_data(self, data_generator):
        dataset = self._get_dataset()
        batch_img, batch_labels, status = dataset.get_a_batch_data(data_generator)
        return batch_img, batch_labels, status

    def _record_test_speed(self, model, start_time, end_time):
        dual_time = end_time - start_time
        speed = self.hyper["batch_size"]/dual_time
        model["test_speed"] = speed
        return model

    def show_hyperparameters(self):
        print("pyperparameters:")
        for key in self.hyper:
            print("%s:%s" % (key, self.hyper[key]))

    def _normalization(self, batch_img):
        return batch_img / 255.

    def _print_train_status(self, model):
        print("epoch:%s steps:%s Train_Loss:%2.5f Train_Acc:%2.5f" % (model["epochs"], model["steps"], model["train_loss"], model["train_accuracy"]))

    def _print_test_status(self, model):
        print("E:%s S:%s Train_Loss:%2.5f Test_Loss:%2.5f Train_Acc:%2.5f Test_Acc:%2.5f gap:%2.5f Train_Speed:%s Test_Speed:%s" % (model["epochs"], model["steps"], model["train_loss"], model["test_loss"], model["train_accuracy"], model["test_accuracy"], model["train_test_gap"], model["train_speed"], model["test_speed"]))
        print("best_epoch:%s best_test_acc:%s" % (model["best_epoch"], model["best_test_accuracy"]))

    def _test_update_model(self, model, avg_loss, accuracy):
        if accuracy > model["best_test_accuracy"]:
            model["best_test_accuracy"] = accuracy
            model["best_epoch"] = model["epochs"]

        model["test_loss"] = avg_loss
        model["test_accuracy"] = accuracy
        model["train_test_gap"] = model["train_accuracy"] - model["test_accuracy"]
        return model

    def _record_model_status(self, model):
        steps_dic = {}
        steps_dic["epochs"] = model["epochs"]
        steps_dic["steps"] = model["steps"]
        steps_dic["train_loss"] = model["train_loss"]
        steps_dic["train_accuracy"] = model["train_accuracy"]
        steps_dic["test_loss"] = model["test_loss"]
        steps_dic["test_accuracy"] = model["test_accuracy"]
        steps_dic["train_test_gap"] = model["train_test_gap"]
        record = model["record"]
        record[model["steps"]] = steps_dic

    def _plot_record(self, model):
        self._plot_a_key(model, "train_loss", "test_loss")
        self._plot_a_key(model, "train_accuracy", "test_accuracy")

    def _plot_a_key(self, model, train_key, test_key):
        record = model["record"]
        train = []
        test = []
        steps = []
        for key in record:
            steps.append([key])
        steps.sort()
        for i in range(len(steps)):
            step_dic = record[steps[i][0]]
            train_value = step_dic[train_key]
            train.append(train_value)
            test_value = step_dic[test_key]
            test.append(test_value)
        train = np.array(train)
        steps = np.array(steps)
        plt.plot(steps, train)
        plt.plot(steps, test)
        plt.show()

    def run_with_epoch(self):
        model = None
        while 1:
            model = self._train_model_with_epochs(model)
            self._test_model(model)
            if model["epochs"] > self.hyper["max_epochs"]:
                break
            if model["epochs"] > model["best_epoch"] + 3:
                print("early stop")
                break
        self._plot_record(model)

    def run_with_steps(self):
        model = None
        data_generator = None
        while 1:
            model, data_generator = self._train_model_with_steps(model, data_generator)
            model = self._test_model(model)
            if model["steps"] > self.hyper["max_steps"]:
                break
            if model["epochs"] > model["best_epoch"] + 3:
                print("early stop")
                break
        self._plot_record(model)
