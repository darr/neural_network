#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : model_base.py
# Create date : 2018-12-25 20:04
# Modified date : 2019-01-03 16:00
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import cifar10_dataset
import mnist_dataset
import func
import pyfile
import time

class ModelBase(object):
    def __init__(self,input_dic):
        super(ModelBase, self).__init__()
        self.hyper = self._get_hyperparameters(input_dic)
        self.output_path = self._get_output_path(input_dic)
        self.output_name = self._get_output_name(input_dic)

    def _get_output_path(self, input_dic):
        fold_name = "%s_%s_%s_%s_%s_%s/%s" % (input_dic["dataset"],
                                        len(input_dic["hidden_layer"]),
                                        input_dic["batch_size"],
                                        input_dic["activation"],
                                        input_dic["hidden_layer"],
                                        input_dic["learn_rate"],
                                        int(time.time()))
        return "./output/%s" % fold_name

    def _get_output_name(self, input_dic):
        file_name = "%s_%s_%s_%s_%s_%s" % (input_dic["dataset"],
                                        len(input_dic["hidden_layer"]),
                                        input_dic["batch_size"],
                                        input_dic["activation"],
                                        input_dic["hidden_layer"],
                                        input_dic["learn_rate"],
                                        )
        return file_name

    def _get_best_status_str(self, model):
        best_str = "best_epoch:%s best_step:%s best_test_acc:%s" % (model["best_epoch"], model["best_step"], model["best_test_accuracy"])
        return best_str

    def _write_best_status(self, model):
        self._write_end_status(self.output_name)
        best_str = self._get_best_status_str(model)
        self._write_end_status(best_str)

    def _write_end_status(self, status_str):
        file_obj = pyfile.open_file("./output/", "output")
        file_obj.write("%s\n" % status_str)
        file_obj.close()

    def _write_status(self, status_str):
        file_obj = pyfile.open_file(self.output_path, self.output_name)
        file_obj.write("%s\n" % status_str)
        file_obj.close()

    def _get_hyperparameters(self, input_dic):
        dic = {}
        dic["activation"] = input_dic["activation"]
        dic["batch_size"] = input_dic["batch_size"]
        dic["learn_rate"] = input_dic["learn_rate"]
        dic["dataset"] = input_dic["dataset"]  # mnist or cifar10
        dic["file_path"] = "./%s_data/" % dic["dataset"]

        dic["epsilon"] = 0.0000001
#        dic["reg_lambda"] = 0.05
        dic["max_steps"] = 300
        dic["train_steps"] = 2
        dic["max_epochs"] = 50
        dic["max_gradient"] = 100.0
        dic["clip_gradient"] = True

        if dic["dataset"] == "mnist":
            dic["input_dim"] = 28*28
            dic["output_dim"] = 10
        elif dic["dataset"] == "cifar10":
            dic["input_dim"] = 32*32*3
            dic["output_dim"] = 10

        dic["early_stop"] = True
        dic["early_stop_condition"] = "step"
        dic["early_stop_gap"] = 50

        lt = []
        lt.append(dic["input_dim"])
        lt.extend(input_dic["hidden_layer"])
        lt.append(dic["output_dim"])

        dic["architecture"] = lt
        return dic

    def _get_dataset(self):
        dic = {}
        dic["file_path"] = "%s" % self.hyper["file_path"]
        dic["label_size"] = 1
        dic["class_nums"] = 10
        if self.hyper["dataset"] == "mnist":
            dic["img_size"] = 28*28
            dataset = mnist_dataset.MnistSet(dic)
        if self.hyper["dataset"] == "cifar10":
            dic["img_size"] = 32*32*3
            dataset = cifar10_dataset.Cifar10Set(dic)
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
        speed = int(self.hyper["batch_size"]/dual_time)
        model["test_speed"] = speed
        return model

    def show_hyperparameters(self):
        hyper_str = "pyperparameters:"
        self._write_status(hyper_str)
        print(hyper_str)
        for key in self.hyper:
            hyper_str = "%s:%s" % (key, self.hyper[key])
            self._write_status(hyper_str)
            print(hyper_str)

    def _normalization(self, batch_img):
        return batch_img / 255.

    def _print_train_status(self, model):
        print("epoch:%s steps:%s Train_Loss:%2.5f Train_Acc:%2.5f" % (model["epochs"], model["steps"], model["train_loss"], model["train_accuracy"]))

    def _print_test_status(self, model):
        status_str = "E:%s S:%s Train_Loss:%2.5f Test_Loss:%2.5f Train_Acc:%2.5f Test_Acc:%2.5f gap:%2.5f Train_Speed:%s Test_Speed:%s" % (model["epochs"], model["steps"], model["train_loss"], model["test_loss"], model["train_accuracy"], model["test_accuracy"], model["train_test_gap"], model["train_speed"], model["test_speed"])
        best_str = "best_epoch:%s best_step:%s best_test_acc:%s" % (model["best_epoch"], model["best_step"], model["best_test_accuracy"])
        self._write_status(status_str)
        self._write_status(best_str)
        print(status_str)
        print(best_str)

    def _test_update_model(self, model, avg_loss, accuracy):
        if accuracy > model["best_test_accuracy"]:
            model["best_test_accuracy"] = accuracy
            model["best_epoch"] = model["epochs"]
            model["best_step"] = model["steps"]

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
        full_path_name = "%s/%s_%s.jpg" % (self.output_path, self.output_name, train_key)
        plt.savefig(full_path_name)
        plt.close()

    def _check_early_stop(self, model):
        if self.hyper["early_stop"]:
            gap = self.hyper["early_stop_gap"]
            if self.hyper["early_stop_condition"] == "step":
                if model["steps"] > model["best_step"] + gap:
                    print("early stop with step")
                    return True
            elif self.hyper["early_stop_conditon"] == "epoch":
                if model["epochs"] > model["best_epoch"] + gap:
                    print("early stop with epoch")
                    return True
        return False

    def _check_max_epochs(self, model):
        if model["epochs"] > self.hyper["max_epochs"]:
            print("max epochs stop")
            return True
        return False

    def _check_stop(self, model):
        if self._check_early_stop(model):
            self._write_best_status(model)
            return True

        if self._check_max_epochs(model):
            self._write_best_status(model)
            return True

        return False

    def run_with_epoch(self):
        model = None
        while 1:
            model = self._train_model_with_epochs(model)
            self._test_model(model)
            if self._check_stop(model):
                break
        self._plot_record(model)

    def run_with_steps(self):
        model = None
        data_generator = None
        while 1:
            model, data_generator = self._train_model_with_steps(model, data_generator)
            model = self._test_model(model)
            if self._check_stop(model):
                break
        self._plot_record(model)
