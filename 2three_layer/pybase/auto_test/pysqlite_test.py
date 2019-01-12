#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pysqlite_test.py
# Create date : 2018-10-16 18:36
# Modified date : 2018-10-19 16:27
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from pybase import pysqlite
from pybase import pylog

def _test_sqlite_table():
    tb_obj = pysqlite.SQLiteTable()
    dic = {'key':9,'content':'test'}
    tb_obj.insert_item_with_dic(dic)
    tb_obj.insert_item(col_names=['key','content'], data_value=[10,'test'])
    tb_obj.insert_item(col_names=['key','content'], data_value=[11,'test'])
    tb_obj.insert_item(col_names=['key','content'], data_value=[12,'test'])
    tb_obj.insert_item(col_names=['key','content'], data_value=[13,'test'])
    tb_obj.insert_item(col_names=['key','content'], data_value=[14,'test'])
    tb_obj.insert_item(col_names=['key','content'], data_value=[15,'test'])

    ret = tb_obj.query_item()
    real_value = len(ret)
    expect_value = 7
    pylog.test_compare_value(real_value, expect_value)

    ret = tb_obj.delete_item("tb_id=4")
    real_value = ret
    expect_value = True
    pylog.test_compare_value(real_value, expect_value)

    ret = tb_obj.query_item()
    real_value = len(ret)
    expect_value = 6
    pylog.test_compare_value(real_value, expect_value)

    ret = tb_obj.update_item(update_str='content="updatewa"', key_str='tb_id=2')
    real_value = ret
    expect_value = True
    pylog.test_compare_value(real_value, expect_value)

    ret = tb_obj._drop_table()
    real_value = ret
    expect_value = True
    pylog.test_compare_value(real_value, expect_value)

    ret = tb_obj.database_name
    os.popen("rm %s.db" % ret)

def test():
    _test_sqlite_table()
