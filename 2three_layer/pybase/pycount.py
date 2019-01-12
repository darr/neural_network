#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pycount.py
# Create date : 2013-06-25 22:24
# Modified date : 2018-10-23 15:54
# Author : DARREN
# Describe : programe action counts
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys

if sys.version > '3':
    from pybase import pyglobal
    from pybase import pylog
else:
    import pyglobal
    import pylog

def get_interface_load_count_dic():
    return pyglobal.interface_load_count

def show_interface_load_count_dic():
    dic = get_interface_load_count_dic()
    pylog.info(dic)

def add_count_with_key(key):
    dic = get_interface_load_count_dic()
    if dic.get(key,''):
        dic[key] = dic.get(key,0) + 1
    else:
        dic[key] = 1

def change_dic():
    dic = get_interface_load_count_dic()
    dic['test'] = "test"
    show_interface_load_count_dic()

def run():
    show_interface_load_count_dic()
    change_dic()
    show_interface_load_count_dic()
    add_count_with_key('contrl_test')
    show_interface_load_count_dic()
    add_count_with_key('contrl_test')
    show_interface_load_count_dic()

if __name__ == "__main__":
    run()

