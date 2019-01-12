#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pysqlite.py
# Create date : 2014-01-07 14:18
# Modified date : 2018-10-20 13:25
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys
import sqlite3

#import pydbbase

if sys.version > '3':
    from pybase import pybaselog
    from pybase import pydbbase
    from pybase import etc
else:
    import pybaselog
    import pydbbase
    import etc

#   if sys.version > '3':
#       import importlib
#       importlib.reload(sys)
#   else:
#       reload(sys)
#       sys.setdefaultencoding('utf8')

def get_charset():
    return etc.DATABASE_CHARSET

def get_table_name():
    return 'pysqlite_tb'

def get_database_table_key_dic():
    return {
        "tb_id":"INTEGER PRIMARY KEY NOT NULL",
        "ct":"INTEGER",
        "ut":"INTEGER",
        "key":"NVARCHAR(32)",
        "content" : "TEXT",
        }

def get_database_name():
    return "./sqlite_test"

def get_database_key_string():
    keylist = []
    db_table_key_dic = get_database_table_key_dic()
    for key in db_table_key_dic:
        key_list_item = '%s %s' %(key, db_table_key_dic[key])
        keylist.append(key_list_item)
    return ','.join(keylist)

def get_create_table_str():
    return "CREATE TABLE IF NOT EXISTS %s(%s)" % (get_table_name(), get_database_key_string())


# pylint: disable=bad-continuation
class SQLiteTable(pydbbase.SQLiteObject):
    def __init__(
                self,
                table_name=get_table_name(),
                database_name=get_database_name(),
                create_table_str=get_create_table_str(),
                charset=get_charset(),
                ):

        super(SQLiteTable, self).__init__()

        self.table_name = table_name
        self.charset = charset
        self.database_name = database_name

        self._create_database()

        self.cursor = self.conn.cursor()
        self.create_table_str = create_table_str
        self._create_table()

    def _create_database(self):
        try:
            db_name = '%s.db' % self.database_name
            self.conn = sqlite3.connect(db_name)
        except Exception as err:
            pybaselog.error(err)
# pylint: enable=bad-continuation
