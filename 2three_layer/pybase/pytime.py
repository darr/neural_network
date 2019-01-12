#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pytime.py
# Create date : 2013-08-24 17:21
# Modified date : 2018-10-20 13:34
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

def get_timestamp():
    return int(time.time())

def get_day():
    return time.strftime("%Y-%m-%d")

def get_today():
    return time.strftime("%Y-%m-%d")
