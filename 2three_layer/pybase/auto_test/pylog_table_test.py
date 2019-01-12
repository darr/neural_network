#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pylog_table_test.py
# Create date : 2018-10-19 14:07
# Modified date : 2018-10-19 17:05
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pylog_table
from pybase import pybaselog
from pybase import pysql


def _get_test_log_dic():
    item_dic = {}
    item_dic['level'] = 'error'
    item_dic['line'] = '125'
    item_dic['func'] = 'function name'
    item_dic['filename'] = 'filename'
    item_dic['time'] = '2016-7-15'
    item_dic['content'] = u'i这个事测试'
    return item_dic

def _test_sqlite_pylog_table():
    tb = pylog_table.PylogTable(db_type=pysql.DB_TYPE_SQLITE)
    test_dic = _get_test_log_dic()
    ret = tb.insert(test_dic)
    tb.drop_table()

def _test_mysql_pylog_table():
    tb = pylog_table.PylogTable(db_type=pysql.DB_TYPE_MYSQL)
    test_dic = _get_test_log_dic()
    ret = tb.insert(test_dic)
    tb.drop_table()

def _test_pylog_table():
    tb = pylog_table.PylogTable()
    item_dic = {}
    item_dic['level'] = 'error'
    item_dic['line'] = '125'
    item_dic['func'] = 'function name'
    item_dic['filename'] = 'filename'
    item_dic['time'] = '2016-7-15'
    item_dic['content'] = u'i这个事测试'
    ret = tb.insert(item_dic)

def test():
    #_test_pylog_table()
    _test_sqlite_pylog_table()
    _test_mysql_pylog_table()
