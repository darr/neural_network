#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pymysqldb_test.py
# Create date : 2018-10-16 23:14
# Modified date : 2018-10-19 13:17
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pymysqldb
from pybase import pylog

def sqlObjectTest():
    tb_obj = pymysqldb.SQLTable()
    dic = {}
    dic['pid'] = 5
    dic['name'] = 'test'
    ret = tb_obj.insert_item_with_dic(dic)
    pylog.test_compare_value(True, ret)
    tb_obj.insert_item(col_names=['pid', 'name'], data_value=[4, 'test'])
    pylog.test_compare_value(True, ret)
    tb_obj.insert_item(col_names=['pid', 'name'], data_value=[3, 'test'])
    pylog.test_compare_value(True, ret)
    tb_obj.insert_item(col_names=['pid', 'name'], data_value=[6, 'test'])
    pylog.test_compare_value(True, ret)
    ret = tb_obj.delete_item("pid=4")
    pylog.test_compare_value(True, ret)
    ret = tb_obj.update_item(update_str='name="update_name"', key_str='pid=3')
    pylog.test_compare_value(True, ret)
    ret = tb_obj.update_item()
    pylog.test_compare_value(False, ret)
    query_content = tb_obj.query_item()
    ret = len(query_content)
    pylog.test_compare_value(3, ret)

#   if query_content:
#       for row in query_content:
#           print(row)
    ret = tb_obj._drop_table()
    pylog.test_compare_value(True, ret)
    ret = tb_obj._drop_database()
    pylog.test_compare_value(True, ret)

def _test_mysql_table():
    sqlObjectTest()

def test():
    _test_mysql_table()
