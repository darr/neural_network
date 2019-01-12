#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pythread_test.py
# Create date : 2018-10-25 17:42
# Modified date : 2018-10-27 13:26
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from pybase import pythread
from pybase import pylog

def _test_get_thread_id():
    pd = pythread.get_thread_id()
    #pylog.info("%s"% pd)

def test():
    _test_get_thread_id()
