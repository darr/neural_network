#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : func.py
# Create date : 2018-12-30 21:33
# Modified date : 2019-01-12 15:55
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import numpy.random as random

def init_W(row, col):
    #np.random.seed(0)
    W = random.randn(row, col)
    #W = np.(row, col),0.1)
    return W

def init_b(col):
    #np.random.seed(0)
    b = np.ones((1, col))
    return b

def get_loss(prob, Y):
    batch_size = Y.shape[0]
    correct_probs = np.dot(prob,Y.T).diagonal()
    correct_logprobs = -log(correct_probs)
    data_loss = np.sum(correct_logprobs)
    loss = 1./batch_size * data_loss
    return loss

def get_comp(prob, Y):
    pre_Y = get_argmax(prob)
    label_lt = get_argmax(Y)
    comp = check_equal(pre_Y,label_lt)
    return comp

def check_equal(X,Y):
    return X == Y

def get_argmax(Y):
    return np.argmax(Y, axis=1)

def get_accuracy(comp, Y):
    return np.flatnonzero(comp).size/Y.shape[0]

def get_correct_number(comp):
    return np.flatnonzero(comp).size

def dot(x,y):
    return np.dot(x,y)

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
    x[x <= 0.0] = 0.0
    x[x > 0.0] = 1.0
    return x

def sig_deirvative(x):
    return sig(x) * (1 - sig(x))

def log(x, epsilon=0.0000001):
    x = x + epsilon
    try:
        y = np.log(x)
        return y
    except Exception as e:
        print(e)
        return None

def clip_gradient(x, max_gradient):
    x[x >= max_gradient] = max_gradient
    x[x <= -max_gradient] = - max_gradient
    return x

def softmax(logits, epsilon=0.0000001):
    logits_max = np.max(logits, axis=1)
    for i in range(logits.shape[0]):
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
