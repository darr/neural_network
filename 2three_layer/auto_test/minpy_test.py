#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : minpy_test.py
# Create date : 2019-01-10 16:39
# Modified date : 2019-01-10 17:46
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import minpy.numpy as np
import minpy.numpy.random as random
from minpy.context import cpu, gpu
import time


def test_creation():
    a = np.array([1,2,3])
    print(a)
    b = np.array([[1,2,3],[2,3,4]])
    print(b)
    a = np.zeros((2,3))
    print(a)
    b = np.ones((2,3))
    print(b)
    c = np.full((2,3),7)
    print(c)
    d = np.empty((2,3))
    print(d)

def test_operations():
    a = np.ones((2,3))
    b = np.ones((2,3))
    c = a + b
    d = - c
    print(d)
    e = np.sin(c**2).T
    print(e)
    f = np.maximum(a,c)
    print(f)

def test_index_slice():
    a = np.arange(6)
    a = np.reshape(a,(3,2))
    print(a[:])
    a[1:2] = -1
    print(a)
    d = np.slice_axis(a, axis=1, begin=1,end=2)
    print(d)

def _randn(l,c):
    return random.randn(l,c)

def test_cpu_gpu(n,s):
    #n = 10
    #s = 512
    with cpu():
        x_cpu = _randn(s,s)
        y_cpu = _randn(s,s)
        for i in range(10):
            z_cpu = np.dot(x_cpu, y_cpu)
        z_cpu.asnumpy()
    t0 = time.time()
    for i in range(n):
        z_cpu = np.dot(x_cpu,y_cpu)
    z_cpu.asnumpy()
    t1 = time.time()
    all_cpu_time = t1 - t0

    with gpu(0):
        x_gpu0 = _randn(s,s)
        y_gpu0 = _randn(s,s)
        for i in range(10):
            z_gpu0 = np.dot(x_gpu0,y_gpu0)
        z_gpu0.asnumpy()
        t2 = time.time()
        for i in range(n):
            z_gpu0 = np.dot(x_gpu0, y_gpu0)
        z_gpu0.asnumpy()
        t3 = time.time()
    all_gpu_time = t3 - t2
    print("run on cpu:%.6f s/iter" % (all_cpu_time / n))
    print("run on gpu:%.6f s/iter" % (all_gpu_time / n))
    print("%s cpu_time/gpu_time:%.6f " % (s,all_cpu_time /all_gpu_time ))


def test():
    #test_creation()
    #test_operations()
    #test_index_slice()
#   for i in range(1,500):
#       test_cpu_gpu(500,i)
    test_cpu_gpu(100,1024)
