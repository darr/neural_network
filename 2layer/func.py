#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : func.py
# Create date : 2018-12-30 21:33
# Modified date : 2018-12-31 14:23
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import numpy as np

def sig(x):
    pos = np.where(x >= 0)
    p = 1.0 / (1 + np.exp(-x[pos]))

    neg = np.where(x < 0)
    n = np.exp(x[neg]) / (1 + np.exp(x[neg]))

    x[pos] = p
    x[neg] = n
    return x

def relu(x):
    return (np.abs(x) + x) / 2.0

def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def sig_deirvative(x):
    return sig(x) * (1 - sig(x))

def log(x, epsilon=0.0000001):
    x = x + epsilon
    return np.log(x)

def softmax(logits, epsilon=0.0000001):
    logits_max = np.max(logits, axis=1)
    for i in range(len(logits)):
        logits[i] = logits[i] - logits_max[i]
    logits = logits + epsilon
    exp_score = np.exp(logits)
    prob = exp_score/np.sum(exp_score, axis=1, keepdims=1)
    return prob

def activation(x, activation_str="relu"):
    activation_dic = {}
    activation_dic["relu"] = relu
    activation_dic["sigmoid"] = sig
    f = activation_dic[activation_str]
    return f(x)

def activation_derivative(x, activation_str="relu"):
    activation_derivative_dic = {}
    activation_derivative_dic["relu"] = relu_derivative
    activation_derivative_dic["sigmoid"] = sig_deirvative
    f = activation_derivative_dic[activation_str]
    return f(x)
#   if "relu" == activation_str:
#       return relu_derivative(x)

#   if "sigmoid" == activation_str:
#       return sig_deirvative(x)
