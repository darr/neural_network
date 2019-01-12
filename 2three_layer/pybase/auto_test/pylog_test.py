#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pylog_test.py
# Create date : 2018-10-07 11:33
# Modified date : 2018-10-09 16:31
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import sys
sys.path.append("..")
import pylog

def test():
    pylog.debug('debug test 测试')
    pylog.info('info test 测试')
    pylog.warning('warning test 测试')
    pylog.error('error test 测试')
    pylog.critical('critical test 测试')
    print(datetime.datetime.now().strftime('%Y-%m-%d'))
