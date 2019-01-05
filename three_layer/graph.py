#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2018-12-30 23:06
# Modified date : 2019-01-03 12:20
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import func

def _record_speed(model, start_time, end_time, hyper_dic):
    dual_time = end_time - start_time
    speed = int(hyper_dic["batch_size"]/dual_time)
    model["train_speed"] = speed
    return model

class NeuralGraph(object):
    def __init__(self):
        super(NeuralGraph, self).__init__()

#   def _weight_decay(self, dW2, dW1, W2, W1):
#       dW2 += self.hyper["reg_lambda"] * W2
#       dW1 += self.hyper["reg_lambda"] * W1
#       return dW1, dW2

    def _init_model(self, hyper_dic, model=None):
        if not model:
            np.random.seed(0)
            architecture = hyper_dic["architecture"]
            weight_lt = []
            bias_lt = []
            for i in range(len(architecture) - 1):
                W = np.random.randn(architecture[i], architecture[i+1])
                b = np.ones((1, architecture[i+1]))
                weight_lt.append(W)
                bias_lt.append(b)

            model = {}
            model["weight_lt"] = weight_lt
            model["bias_lt"] = bias_lt
            model["steps"] = 0
            model["epochs"] = 1
            model["record"] = {}
            model["best_test_accuracy"] = 0.0
            model["best_epoch"] = 1
            model["best_step"] = 1
        return model

    def _normalization(self, batch_img):
        return batch_img / 255.

    def forward_propagation(self, model, X, Y, hyper_dic):
        activation_str = hyper_dic["activation"]
        architecture = hyper_dic["architecture"]
        epsilon = hyper_dic["epsilon"]
        model = self._init_model(hyper_dic, model)
        weight_lt = model["weight_lt"]
        bias_lt = model["bias_lt"]

        linear_output_lt = []
        activation_output_lt = []
        model["linear_output_lt"] = linear_output_lt
        model["activation_output_lt"] = activation_output_lt
        batch_size = hyper_dic["batch_size"]
        activation_output_lt.append(X)

        a = X
        Z = None

        for i in range(len(architecture) - 2):
            Z = np.dot(a, weight_lt[i]) + bias_lt[i]
            model["linear_output_lt"].append(Z)
            a = func.activation(Z, activation_str)
            model["activation_output_lt"].append(a)

        Z = np.dot(a, weight_lt[len(architecture) - 2]) + bias_lt[len(architecture) - 2]
        model["linear_output_lt"].append(Z)

        prob = func.softmax(Z)

        correct_probs = prob[range(batch_size), np.argmax(Y, axis=1)]
        correct_logprobs = -func.log(correct_probs)

        data_loss = np.sum(correct_logprobs)
        loss = 1./batch_size * data_loss

        pre_Y = np.argmax(prob, axis=1)
        comp = pre_Y == np.argmax(Y, axis=1)
        accuracy = len(np.flatnonzero(comp))/Y.shape[0]

        return model, prob, loss, accuracy, comp

    def backward_propagation(self, model, prob, X, Y, hyper_dic):
        activation_str = hyper_dic["activation"]
        learn_rate = hyper_dic["learn_rate"]
        architecture = hyper_dic["architecture"]
        max_gradient = hyper_dic["max_gradient"]
        clip_gradient = hyper_dic["clip_gradient"]

        weight_lt = model["weight_lt"]
        linear_output_lt = model["linear_output_lt"]
        activation_output_lt = model["activation_output_lt"]

        dY_pred = prob - Y
        d_out = dY_pred

        if not len(linear_output_lt) == len(activation_output_lt) == len(weight_lt):
            print("list len is not the same")
        Z = linear_output_lt[len(linear_output_lt) - 1]
        a = activation_output_lt[len(linear_output_lt) - 1]
        W = weight_lt[len(linear_output_lt) - 1]

        for i in range(len(linear_output_lt)-1, -1, -1):
            a = activation_output_lt[i]
            W = weight_lt[i]

            dW = np.dot(a.T, d_out)
            if clip_gradient:
                dW = func.clip_gradient(dW, max_gradient)
            da = np.dot(d_out, W.T)
            weight_lt[i] += -learn_rate * dW
            if i > 0:
                Z = linear_output_lt[i - 1]
                dadZ = func.activation_derivative(Z, activation_str)
                d_out = da * dadZ

        return model

    def core_graph(self, model, X, Y, hyper_dic):
        start_time = time.time()
        X = self._normalization(X)
        model, prob, loss, accuracy, comp = self.forward_propagation(model, X, Y, hyper_dic)
        model["train_loss"] = loss
        model["train_accuracy"] = accuracy
        model = self.backward_propagation(model, prob, X, Y, hyper_dic)
        end_time = time.time()
        model = _record_speed(model, start_time, end_time, hyper_dic)
        return model
