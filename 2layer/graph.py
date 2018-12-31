#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2018-12-30 23:06
# Modified date : 2018-12-31 16:01
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
    speed = hyper_dic["batch_size"]/dual_time
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
            #np.random.seed(0)
            architecture = hyper_dic["architecture"]
            W1 = np.random.randn(architecture[0], architecture[1])
            b1 = np.ones((1, architecture[1]))
            W2 = np.random.randn(architecture[1], architecture[2])
            b2 = np.ones((1, architecture[2]))

            model = {}
            model["W1"] = W1
            model["b1"] = b1
            model["W2"] = W2
            model["b2"] = b2
            model["steps"] = 1
            model["epochs"] = 0
            model["record"] = {}
            model["best_test_accuracy"] = 0.0
            model["best_epoch"] = 0
        return model

    def _normalization(self, batch_img):
        return batch_img / 255.

    def forward_propagation(self, model, X, Y, hyper_dic):
        activation_str = hyper_dic["activation"]
        model = self._init_model(hyper_dic, model)
        W1 = model["W1"]
        b1 = model["b1"]
        W2 = model["W2"]
        b2 = model["b2"]

        X = self._normalization(X)
        Z1 = np.dot(X, W1)+b1
        a1 = func.activation(Z1, activation_str)
        logits = np.dot(a1, W2)+b2
        prob = func.softmax(logits)

        correct_probs = prob[range(X.shape[0]), np.argmax(Y, axis=1)]
        correct_logprobs = - func.log(correct_probs)

        data_loss = np.sum(correct_logprobs)
        loss = 1./X.shape[0] * data_loss

        pre_Y = np.argmax(prob, axis=1)
        comp = pre_Y == np.argmax(Y, axis=1)
        accuracy = len(np.flatnonzero(comp))/Y.shape[0]

        return model, prob, a1, Z1, loss, accuracy, comp

    def backward_propagation(self, model, prob, X, Y, a1, Z1, hyper_dic):
        activation_str = hyper_dic["activation"]
        learn_rate = hyper_dic["learn_rate"]

        #W1 = model["W1"]
        W2 = model["W2"]
        dY_pred = prob - Y
        dW2 = np.dot(a1.T, dY_pred)
        da1 = np.dot(dY_pred, W2.T)
        dadZ = func.activation_derivative(Z1, activation_str)
        dZ1 = da1 * dadZ
        dW1 = np.dot(X.T, dZ1)
        #dW1,dW2 = self.weight_decay(dW2,dW1,W2,W1)
        model["W2"] += -learn_rate * dW2
        model["W1"] += -learn_rate * dW1

        return model


    def core_graph(self, model, X, Y, hyper_dic):
        start_time = time.time()
        model, prob, a1, Z1, loss, accuracy, comp = self.forward_propagation(model, X, Y, hyper_dic)
        model["train_loss"] = loss
        model["train_accuracy"] = accuracy
        model = self.backward_propagation(model, prob, X, Y, a1, Z1, hyper_dic)
        end_time = time.time()
        model = _record_speed(model, start_time, end_time, hyper_dic)
        return model
