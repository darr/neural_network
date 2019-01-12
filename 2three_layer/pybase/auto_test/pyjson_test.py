#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyjson_test.py
# Create date : 2018-10-10 00:08
# Modified date : 2018-10-10 00:25
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import os
#import sys

from pybase import pyjson
from pybase import pylog


def _json_ok_test():
    test_str = "json ok"
    json_con = pyjson.json_ok(test_str)
    json_dic = pyjson.json_to_dict(json_con)
    pylog.test_compare_value(json_dic["request_method"],"GET")
    pylog.test_compare_value(json_dic["data"],test_str)
    pylog.test_compare_value(json_dic["result"],"ok")

def _json_fail_test():
    test_str = "json fail"
    json_con = pyjson.json_fail(test_str)
    json_dic = pyjson.json_to_dict(json_con)
    pylog.test_compare_value(json_dic["request_method"],"GET")
    pylog.test_compare_value(json_dic["data"],test_str)
    pylog.test_compare_value(json_dic["result"],"fail")

def _json_method_not_support_test():
    test_str = "method not support"
    json_con = pyjson.json_method_not_support(test_str)
    json_dic = pyjson.json_to_dict(json_con)
    pylog.test_compare_value(json_dic["request_method"],"GET")
    pylog.test_compare_value(json_dic["data"],test_str)
    pylog.test_compare_value(json_dic["result"],"fail")

def test():
    _json_ok_test()
    _json_fail_test()
    _json_method_not_support_test()

