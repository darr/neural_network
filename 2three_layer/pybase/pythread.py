#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pythread.py
# Create date : 2018-10-25 17:36
# Modified date : 2018-10-26 18:01
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import threading

def get_thread_id():
    return threading.current_thread().ident

def start_thread():
    t1 = threading.Thread(target=run_thread, args=(5,))
    return t1
