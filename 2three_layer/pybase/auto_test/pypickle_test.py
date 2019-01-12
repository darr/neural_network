#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pypickle_test.py
# Create date : 2018-10-15 18:43
# Modified date : 2018-10-15 22:58
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pypickle
from pybase import pylog


def pickle_test():
    dic = {
            "key" : "111",
            "id" : 222,
            "value" : 333,
            "name" : "nihao",
            "age" : 18,
            }
    file_name = 'hello'
    ret = pypickle.save(dic, file_name)
    pylog.test_compare_value(True,ret)
    if ret == True:
        dic = pypickle.read(file_name)
        pylog.test_compare_value(18, dic["age"])

def run():
    pickle_test()

def test():
    run()

