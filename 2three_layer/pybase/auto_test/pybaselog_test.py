#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pybaselog_test.py
# Create date : 2018-10-09 17:22
# Modified date : 2018-10-09 17:23
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pybaselog

def test():
    pybaselog.debug('debug test 测试')
    pybaselog.info('info test 测试')
    pybaselog.warning('warning test 测试')
    pybaselog.error('error test 测试')
    pybaselog.critical('critical test 测试')
