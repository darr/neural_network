#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pylog_table.py
# Create date : 2016-07-22 11:39
# Modified date : 2018-10-20 14:56
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

from __future__ import division
from __future__ import print_function

import sys
#import pydb
#import pysql
#import pybaselog

if sys.version > '3':
    from pybase import etc
    from pybase import pyobject
    from pybase import pydb
    from pybase import pysql
    from pybase import pybaselog
else:
    import etc
    import pyobject
    import pydb
    import pysql
    import pybaselog

def get_table_key_dic():
    return {
        "id":pysql.INT_PRIMARY_KEY_AUTO_INCREMENT,
        "ct" : pysql.TYPE_INT,
        "ut" : pysql.TYPE_INT,
        "level": pysql.TYPE_TEXT,
        "line": pysql.TYPE_TEXT,
        "func": pysql.TYPE_TEXT,
        "filename": pysql.TYPE_TEXT,
        "time": pysql.TYPE_TEXT,
        "content": pysql.TYPE_TEXT,
        "log_key": pysql.TYPE_TEXT,
        }

class PylogTable(pyobject.BaseObject):
    def __init__(self, db_type=pysql.DB_TYPE_SQLITE):
        #super(pyobject.BaseObject, self).__init__()
        super(PylogTable, self).__init__()
        if db_type == pysql.DB_TYPE_SQLITE:
            self.log_table = self._create_sqlite_table()
        elif db_type == pysql.DB_TYPE_MYSQL:
            self.log_table = self._create_mysql_table()
        else:
            pybaselog.condition_value_is_null()

    def _create_sqlite_table(self):
        log_table = pydb.PyTable()
        db_name = etc.PYLOG_DATABASE_NAME
        db_path = etc.PYLOG_DATABASE_PATH
        tb_name = etc.PYLOG_TABLE_NAME
        table_dic = get_table_key_dic()
        log_table.create_sqlite_table_with_dic(table_dic, db_name, tb_name, db_path)
        return log_table

    def _create_mysql_table(self):
        log_table = pydb.PyTable()
        db_name = etc.PYLOG_DATABASE_NAME
        tb_name = etc.PYLOG_TABLE_NAME
        db_host = "localhost"
        user_name = "dluser"
        password = "888888"
        dic = get_table_key_dic()
        log_table.create_mysql_table_with_dic(dic, db_name, tb_name, db_host, user_name, password)
        return log_table

    def insert(self, item_dic):
        return self.log_table.insert(item_dic)

    def drop_table(self):
        return self.log_table.drop_table()
