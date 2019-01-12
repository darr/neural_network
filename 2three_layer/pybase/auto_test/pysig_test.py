#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pysig_test.py
# Create date : 2018-10-25 15:46
# Modified date : 2018-10-26 16:40
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import time
import signal

from pybase import pylog
from pybase import pysig

def _test_handler_to_signal():
    signal.signal(signal.SIGALRM, pysig.handler_to_signal)
    signal.alarm(5)
    signal.pause()

def _test_set_signal_SIGALRM():
    pysig.set_signal_SIGALRM()

def _test_signals():
    pysig.add_all_signal_funcs()
    #signal.pause()

def test():
    #_test_handler_to_signal()
    _test_signals()
