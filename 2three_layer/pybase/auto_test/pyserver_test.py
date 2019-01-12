#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyserver_test.py
# Create date : 2018-10-19 18:14
# Modified date : 2018-10-26 16:30
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pyserver
from pybase import pylog

def _test_dlserver():
    port = "8789"
    name = "dlserver_test"
    dlServer = pyserver.DLServer(proc_name=name,listen_port=port)
    dlServer.work()

def test():
    #_test_dlserver()
    pass
