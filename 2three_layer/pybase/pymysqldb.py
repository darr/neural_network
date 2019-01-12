#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pymysqldb.py
# Create date : 2013-08-05 12:13
# Modified date : 2018-10-20 13:34
# Author : DARREN
# Describe : cotrl mysqldb
# Email : lzygzh@126.com
#####################################
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if sys.version > '3.4':
    import pymysql as MySQLdb
else:
    import MySQLdb

if sys.version > '3':
    from pybase import pydbbase
    from pybase import etc
else:
    import pydbbase
    import etc

def get_user_name():
    return etc.DATABASE_USE

def get_host():
    return etc.DATABASE_HOST

def get_password():
    return etc.DATABASE_PASSWORD

def get_charset():
    return etc.DATABASE_CHARSET

def get_dataname():
    return etc.DATABASE_NAME

def get_table_name():
    return etc.DATABASE_TABLE_NAME

def get_table_create_str():
    return etc.TABLE_TEST


# pylint: disable=bad-continuation
class SQLTable(pydbbase.MySQLObject):
    def __init__(
                self,
                table_name=get_table_name(),
                database_name=get_dataname(),
                host=get_host(),
                user=get_user_name(),
                password=get_password(),
                charset=get_charset(),
                create_table_str=get_table_create_str()
                ):

        super(SQLTable, self).__init__()

        self.table_name = table_name
        self.charset = charset
        self.database_name = database_name

        self.host = host
        self.user = user
        self.password = password
        self.create_str = self._get_create_str()

        self._create_database()
        self.conn = self._get_database_connect()

        self.cursor = self.conn.cursor()
        self.create_table_str = create_table_str
        self._create_table()

    def _create_database(self):
        conn = MySQLdb.connect(host=self.host,
                               user=self.user,
                               passwd=self.password,
                               charset=self.charset
                               )
        cursor = conn.cursor()
        self._cursor_execute(cursor)
        self._close_cursor(cursor)

    def _get_database_connect(self):
        conn = MySQLdb.connect(host=self.host,
                               user=self.user,
                               passwd=self.password,
                               charset=self.charset,
                               db=self.database_name
                               )
        return conn
# pylint: enable=bad-continuation
