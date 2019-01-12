#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pydb_test.py
# Create date : 2018-10-18 22:13
# Modified date : 2018-10-20 14:56
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from pybase import pydb
from pybase import pylog
from pybase import pysql
from pybase import pytime
from pybase import etc

######################test_code######################
def get_table_key_dic():
    return {
        "tb_id":pysql.INT_PRIMARY_KEY_AUTO_INCREMENT,
        "ky":pysql.TYPE_TEXT,
        "con" : pysql.TYPE_TEXT,
        "ut" : pysql.TYPE_INT,
        }

def get_test_dic():
    dic = {
            "ky":"baidu",
            "con":"http://ww1.sinaimg.cn/large/005JS8khjw1enuy99dfsfj30cs11uju7.jpg",
            }
    return  dic

def test_create_sqlite_table_with_dic():
    ptb = pydb.PyTable()
    db_name = "pydb_test"
    db_path = "./"
    tb_name = "pydb_tb_test"
    table_dic = get_table_key_dic()
    ptb.create_sqlite_table_with_dic(table_dic,db_name,tb_name,db_path)
    dic = get_test_dic()
    dic["ut"] = pytime.get_timestamp()
    ptb.insert(dic)
    ret = ptb.query_item()
    ptb.drop_table()
    ret = ptb.db_full_name
    os.popen("rm %s.db" % ret)
    #ptb.drop_database()
    #pylog.info(ret)

def test_create_mysql_table_with_dic():
    ptb = pydb.PyTable()
    db_name = etc.DATABASE_NAME
    tb_name = "pydb_tb_test"
    db_host = "localhost"
    user_name = "dluser"
    password = "888888"
    dic = get_table_key_dic()
    ptb.create_mysql_table_with_dic(dic,db_name,tb_name,db_host,user_name,password)
    dic = get_test_dic()
    dic["ut"] = pytime.get_timestamp()
    ptb.insert(dic)
    ret = ptb.query_item()
    ptb.drop_table()
    #ptb.drop_database()
    #pylog.info(ret)

def common_test():
    test_create_sqlite_table_with_dic()
    test_create_mysql_table_with_dic()

def test():
    common_test()
