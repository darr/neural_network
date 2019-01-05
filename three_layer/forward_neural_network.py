#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : forward_neural_network.py
# Create date : 2018-12-25 20:04
# Modified date : 2019-01-03 11:16
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import time

import numpy as np

import func
import model_base
import graph

class DeepModel(model_base.ModelBase):
    def __init__(self, input_dic):
        super(DeepModel, self).__init__(input_dic)
        self.graph = graph.NeuralGraph()

    def _train_model_with_epochs(self, model=None):
        data_generator = self._get_train_generator()
        while 1:
            X, Y, status = self._get_a_batch_data(data_generator)
            if status == False:
                model['epochs'] += 1
                break
            model = self.graph.core_graph(model, X, Y, self.hyper)
            model["steps"] += 1
            if model["steps"] % self.hyper["train_steps"] == 0:
                self._print_train_status(model)

        return model

    def _train_model_with_steps(self, model=None, data_generator=None):
        if data_generator == None:
            data_generator = self._get_train_generator()
        while 1:
            X, Y, status = self._get_a_batch_data(data_generator)
            if status == False:
                data_generator = self._get_train_generator()
                model['epochs'] += 1
            model = self.graph.core_graph(model, X, Y, self.hyper)

            model["steps"] += 1
            if model["steps"] % self.hyper["train_steps"] == 0:
                break

        return model, data_generator

    def _test_model(self, model):
        data_generator = self._get_test_generator()
        count = 1
        all_correct_numbers = 0
        all_loss = 0.0

        while count:
            X, Y, status = self._get_a_batch_data(data_generator)
            if status == False:
                break
            start_time = time.time()
            X = self.graph._normalization(X)
            model, prob, loss, accuracy, comp = self.graph.forward_propagation(model, X, Y, self.hyper)
            end_time = time.time()
            model = self._record_test_speed(model, start_time, end_time)
            all_loss += loss
            all_correct_numbers += len(np.flatnonzero(comp))
            count += 1

        avg_loss = all_loss / count
        accuracy = all_correct_numbers / (count * self.hyper["batch_size"])
        self._test_update_model(model, avg_loss, accuracy)
        self._print_test_status(model)
        self._record_model_status(model)
        return model
