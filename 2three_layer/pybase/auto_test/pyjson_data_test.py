#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyjson_data_test.py
# Create date : 2018-10-10 00:00
# Modified date : 2018-10-10 00:07
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pyjson_data
from pybase import pylog


def test():
    pylog.test_compare_value(pyjson_data.JSON_TYPE_OK, "ok")
    pylog.test_compare_value(pyjson_data.JSON_TYPE_FAIL, "fail")
    pylog.test_compare_value(pyjson_data.JSON_KEY_RESULT, "result")
    pylog.test_compare_value(pyjson_data.JSON_KEY_DATA, "data")
    pylog.test_compare_value(pyjson_data.JSON_KEY_STATE, "state")
    pylog.test_compare_value(pyjson_data.REQUEST_METHOD, "request_method")
    pylog.test_compare_value(pyjson_data.REQUEST_METHOD_NOT_SUPPORT, "http method not support")
    pylog.test_compare_value(pyjson_data.REQUEST_GET, "GET")
    pylog.test_compare_value(pyjson_data.REQUEST_POST, "POST")

    pylog.test_compare_value(pyjson_data.NO_DESCRIPTION, "no description")
