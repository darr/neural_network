#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : app_test.py
# Create date : 2018-10-07 12:48
# Modified date : 2019-01-12 13:44
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

from pybase import pylinux
from pybase import pylog
from pybase import pymodule
from pybase import pylinux
from pybase import pyprocess
from pybase import pythread
from pybase import pysig

from auto_test import minpy_test
from auto_test import numpy_test

def show_env():
    print(pylinux.get_system_name_version())
    print(pylinux.get_platform_unname()[3])
    print(pylinux.get_architecture())
    print("Python:%s" % sys.version)
    print("pid:%s" % pyprocess.get_current_pid())
    print("thread_id:%s" % pythread.get_thread_id())

def get_env():
    pymodule.write_all_modules()
    pymodule.get_all_modules()
    pylinux.get_linux_apps_list()
    pylinux.get_sys_info()

def add_signal_funcs():
    pysig.add_all_signal_funcs()

def run():
    show_env()
    get_env()
    minpy_test.test()
    numpy_test.test()

if __name__ == '__main__':
    run()



