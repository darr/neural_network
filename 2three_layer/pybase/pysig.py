#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pysig.py
# Create date : 2018-10-25 15:43
# Modified date : 2018-10-26 16:40
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys
import signal

if sys.version > '3':
    from pybase import pylog
else:
    import pylog

SIG_DIC = {
        6:"SIGABRT",
        14:"SIGALRM",
        7:"SIGBUS",
        17:"SIGCHLD",
        #17:"SIGCLD"
        18:"SIGCONT",
        8:"SIGFPE",
        1:"SIGHUP",
        4:"SIGILL",
        2:"SIGINT",
        29:"SIGIO",
        6:"SIGIOT",
        9:"SIGKILL",
        13:"SIGPIPE",
        29:"SIGPOLL",
        27:"SIGPROF",
        39:"SIGPWR",
        3:"SIGQUIT",
        64:"SIGRTMAX",
        34:"SIGRTMIN",
        11:"SIGSEGV",
        19:"SIGSTOP",
        31:"SIGSYS",
        15:"SIGTERM",
        5:"SIGTRAP",
        20:"SIGTSTP",
        21:"SIGTTIN",
        22:"SIGTTOU",
        23:"SIGURG",
        10:"SIGUSR1",
        12:"SIGUSR2",
        26:"SIGVTALRM",
        28:"SIGWINCH",
        24:"SIGXCPU",
        25:"SIGXFSZ",
}

SIG_LIST = [
        "SIGABRT",
        "SIGALRM",
        "SIGBUS",
        "SIGCHLD",
        "SIGCONT",
        "SIGFPE",
        "SIGHUP",
        "SIGILL",
        "SIGINT",
        "SIGIO",
        "SIGIOT",
        #"SIGKILL",
        "SIGPIPE",
        "SIGPOLL",
        "SIGPROF",
        "SIGPWR",
        "SIGQUIT",
        "SIGRTMAX",
        "SIGRTMIN",
        "SIGSEGV",
        #"SIGSTOP",
        "SIGSYS",
        "SIGTERM",
        "SIGTRAP",
        "SIGTSTP",
        "SIGTTIN",
        "SIGTTOU",
        "SIGURG",
        "SIGUSR1",
        "SIGUSR2",
        "SIGVTALRM",
        "SIGWINCH",
        "SIGXCPU",
        "SIGXFSZ",
        ]

def handler_to_signal(sig_num,frame):
    sig_name = SIG_DIC[sig_num]
    pylog.error("receive signal:%s, then exit" % sig_name)
    exit()

for sig_name in SIG_LIST:
    exec('''
def set_signal_%s(handler=None):
    if not handler:
        handler = handler_to_signal
    signal.signal(signal.%s, handler)''' %(sig_name,sig_name))

def add_all_signal_funcs():
    for sig_name in SIG_LIST:
        exec('''set_signal_%s()''' %(sig_name))
