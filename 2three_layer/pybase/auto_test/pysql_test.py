#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pysql_test.py
# Create date : 2018-10-16 22:32
# Modified date : 2018-10-19 11:11
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pysql
from pybase import pylog

def get_table_key_dic():
    return {
        "tb_id":pysql.INT_PRIMARY_KEY_AUTO_INCREMENT,
        "key":pysql.TYPE_TEXT,
        "content" : pysql.TYPE_TEXT,
        "ut" : pysql.TYPE_INT,
        }

def _test_mysql_create_str():
    tb_name = "tb_test"
    dic = get_table_key_dic()
    db_type = pysql.DB_TYPE_MYSQL
    ret = pysql.get_table_create_str_from_dic(dic, tb_name, db_type)
    #pylog.info(ret)

def _test_sqlite_create_str():
    tb_name = "tb_test"
    dic = get_table_key_dic()
    db_type = pysql.DB_TYPE_SQLITE
    ret = pysql.get_table_create_str_from_dic(dic, tb_name, db_type)
    #pylog.info(ret)

def test():
    _test_mysql_create_str()
    _test_sqlite_create_str()

