#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyprocess.py
# Create date : 2016-10-15 06:11
# Modified date : 2018-10-26 16:34
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

from __future__ import division
from __future__ import print_function

import os
import sys
import re
from subprocess import check_output

if sys.version > '3':
    import subprocess as commands
    from pybase import pylog
    from pybase import pyre
else:
    import commands
    import pylog
    import pyre

# pylint: disable=bad-continuation
def _get_progress_pid(progress_name):
    res = commands.getstatusoutput('ps aux|grep "' + str(progress_name) + str('"'))[1].split('\n')
    p = re.compile(r'\s+')
    l = ""
    for item in res:
        l = p.split(item)
        info = {'user':l[0],
               'pid':l[1],
               'cpu':l[2],
               'mem':l[3],
               'vsa':l[4],
               'rss':l[5],
               'tty':l[6],
               'stat':l[7],
               'start':l[8],
               'time':l[9],
               'command':' '.join(l[10:]).strip()}
        if not pyre.re_search("ps aux|grep ", info['command']):
            return l[1]
    return False
# pylint: enable=bad-continuation


def _kill_progress(pid):
    res = commands.getstatusoutput('kill -9 %s' % pid)[1].split('\n')
    return res

#####interface########################################################

def get_pid_lt_with_name(name):
    return map(int,check_output(["pidof",name]).split())

def get_current_pid():
    return os.getpid()

def close_progress(progress_name):
    pid = _get_progress_pid(progress_name)
    if not pid:
        return False
    while(not pid):
        _kill_progress(pid)
        pid = _get_progress_pid(progress_name)
        if not pid:
            break
    return True

def check_progress(progress_name):
    pid = _get_progress_pid(progress_name)
    return pid

def start_progress(progress_name):
    pid = _get_progress_pid(progress_name)
    if not pid:
        ret = os.system("nohup python %s > /dev/null 2>&1 &" % progress_name)
        if ret == 0:
            return True
        else:
            pylog.error("progressname:%s status:%s", (progress_name, ret))
    return False


def restart_python_progress(progress_name):
    c_ret = close_progress(progress_name)
    s_ret = start_progress(progress_name)
    if c_ret and s_ret:
        return True
    return False

def nohup_start_python_progress(progress_name):
    return start_progress(progress_name)
