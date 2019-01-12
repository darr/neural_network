#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pytime_test.py
# Create date : 2018-10-09 17:47
# Modified date : 2018-10-16 23:12
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from pybase import pytime
from pybase import pylog

def _show_time():
    print(time.localtime())
    print(time.localtime(1304575584.1361799))
    print(time.gmtime())
    print(time.time())
    print(time.mktime(time.localtime()))
    print(time.asctime())
    print(time.ctime(time.time()))
    print(time.ctime(1304579615))
    print(time.strftime("%Y-%m-%d %X",time.localtime()))
    print(time.strftime("%Y-%m-%d %X"))
    print(time.strptime('2011-05-05 16:37:06','%Y-%m-%d %X'))

def func_test():
    print("get_timestamp: %s" % pytime.get_timestamp())

def clock_test():
    time.sleep(1)
    print("clock1: %s" % time.clock())
    time.sleep(1)
    print("clock2: %s" % time.clock())
    time.sleep(1)
    print("clock3: %s" % time.clock())

def _test_get_timestamp():
    ret = pytime.get_timestamp()

def _test_get_day():
    ret = pytime.get_day()

def _test_get_today():
    ret = pytime.get_today()

def common_test():
    #_show_time()
    #func_test()
    pass

def run():
    common_test()

def test():
    _test_get_timestamp()
    _test_get_day()
    _test_get_today()

