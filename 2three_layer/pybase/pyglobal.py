#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyglobal.py
# Create date : 2013-06-25 22:19
# Modified date : 2018-10-09 17:47
# Author : DARREN
# Describe : save global vars
# Email : lzygzh@126.com
#####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

interface_load_count = {}

def isPython3():
    import sys
    if sys.version > '3':
        return True
    return False
