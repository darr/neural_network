#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyhttp_test.py
# Create date : 2018-10-16 12:38
# Modified date : 2018-10-16 12:48
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pyhttp
from pybase import pylog

def _test_http_get():
    url = "http://www.baidu.com"
    is_sucess, ret = pyhttp.http_get(url)
    pylog.test_compare_value(is_sucess,True)

def _test_http_post():
    #url = "http://106.187.39.151:8181/fun/insert"
    url = "http://ip:port/app/insert"
    data = {}
    data["title"] = "title"
    data["lnk"] = "lnk"
    data["type"] = 1
    data["level"] = 1
    data["channel"] ="your channel"
    data = pyjson.dict_to_json(data)
    pyhttp.http_post(url,data)

def test():
    _test_http_get()
    #_test_http_post()
