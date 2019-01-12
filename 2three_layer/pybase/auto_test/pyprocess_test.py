#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyprocess_test.py
# Create date : 2018-10-16 12:52
# Modified date : 2018-10-27 13:27
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from pybase import pyprocess
from pybase import pylog

def _test_nohup_start_python_progress():
    progress_name = "yourname"
    ret = pyprocess.nohup_start_python_progress(progress_name)
    pylog.test_compare_value(ret, True)

def _test_restart_python_progress():
    progress_name = "yourname"
    ret = pyprocess.restart_python_progress(progress_name)
    #pylog.test_compare_value(ret, False)

def _test_close_progress():
    progress_name = "yourname"
    ret = pyprocess.close_progress(progress_name)
    #pylog.test_compare_value(ret, False)

def _test_get_current_pid():
    pid = pyprocess.get_current_pid()
    #pylog.info(pid)

def _test_get_pid_lt_with_name():
    name = "chrome"
    pid_lt = pyprocess.get_pid_lt_with_name(name)
    #pylog.info(pid_lt)

def test():
    _test_nohup_start_python_progress()
    _test_restart_python_progress()
    _test_close_progress()
    _test_get_current_pid()
    _test_get_pid_lt_with_name()
