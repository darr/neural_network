#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pydbbase.py
# Create date : 2018-10-19 11:18
# Modified date : 2018-12-24 20:02
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from abc import abstractmethod

from pybase import pyobject
from pybase import pybaselog

#   if sys.version > '3':
#       from pybase import pyobject
#       from pybase import pybaselog
#   else:
#       import pyobject
#       import pybaselog


class DBObject(pyobject.BaseObject):
    def __init__(self):
#        super(pyobject.BaseObject, self).__init__()
        super(DBObject, self).__init__()

    def __del__(self):
        self._close_cursor(self.cursor)
        self.conn.close()

    def _close_cursor(self, cur):
        cur.close()

    def _create_table(self):
        try:
            self.cursor.execute(self.create_table_str)
        except Exception as err:
            pybaselog.error(err)
            return False
        return True

    def _operate_database(self, sql_str):
        try:
            self.cursor.execute(sql_str)
        except Exception as err:
            pybaselog.error(err)
            return False
        return True

    def _get_format_insert_str(self, col_names, format_str='?'):
        value_str = ''
        clo_str = ''
        for clo in col_names:
            if clo_str:
                clo_str = clo_str + ', ' + clo
                value_str = value_str + ', ' + format_str
            else:
                clo_str = clo
                value_str = format_str
        return clo_str, value_str

    def insert_item_with_dic(self, dic):
        key_lt = []
        value_lt = []
        for key in dic:
            key_lt.append(key)
#            value_lt.append(str(dic[key]))
            value_lt.append(dic[key])

        return self.insert_item(key_lt, value_lt)

    @abstractmethod
    def _get_insert_str(self, col_names):
        '''abstract method'''

    def insert_item(self, col_names, data_value):
        clom_str, value_str = self._get_insert_str(col_names)
        sql = "insert into %s(%s) values(%s)" % (self.table_name, clom_str, value_str)
        try:
            self.cursor.execute(sql, data_value)
            self.conn.commit()
        except Exception as err:
            pybaselog.error("error:%s content:%s,sql:%s,table_create_str:%s"%(err, data_value, sql, self.create_table_str))
        return self.cursor.lastrowid

    def _get_update_str(self, update_str='', key_str=''):
        if not update_str or not key_str:
            return False

        sql = "update %s set %s where %s " % (self.table_name, update_str, key_str)
        return sql

    def _get_delete_str(self, delete_condition_str=''):
        if not delete_condition_str:
            pybaselog.warning("delete_condition_str is none")
            return False

        sql = "delete from %s where %s " % (self.table_name, delete_condition_str)
        return sql

    def execute_sql(self, sql_str):
        return self._operate_database(sql_str)

    def query_count(self, search_content="count(*)", search_condition=""):
        if search_condition:
            sql = "select %s from %s where %s" % (search_content, self.table_name, search_condition)
        else:
            sql = "select %s from %s" % (search_content, self.table_name)

        if self._operate_database(sql):
            return self.cursor.fetchall()
        return False

    def query_item(self, search_content='*', search_condition='', page='', page_size='', order_by="order by ut desc"):
        page_str = ""
        if page and page_size:
            #page_str = "limit %s offset %s" %(page_size,(int(page) - 1) * int(page_size))
            page_str = "limit %s, %s" %((int(page) - 1) * int(page_size), page_size)
        if search_condition:
            sql = "select %s from %s where %s %s %s" % (search_content, self.table_name, search_condition, order_by, page_str)
        else:
            sql = "select %s from %s  %s %s" % (search_content, self.table_name, order_by, page_str)
        if self._operate_database(sql):
            return self.cursor.fetchall()
        return False

    def update_item(self, update_str='', key_str=''):
        sql = self._get_update_str(update_str, key_str)
        if not sql:
            return False
        ret = self._operate_database(sql)
        self.conn.commit()
        return ret

    def delete_item(self, delete_condition_str):
        sql = self._get_delete_str(delete_condition_str)
        if not sql:
            return False
        ret = self._operate_database(sql)
        self.conn.commit()
        return ret

    def _drop_table(self):
        return self._operate_database("drop table %s"%self.table_name)

    def _drop_database(self):
        return self._operate_database("drop database %s"%self.database_name)


class SQLiteObject(DBObject):
    def __init__(self):
        super(SQLiteObject, self).__init__()

    def _get_insert_str(self, col_names):
        return super(SQLiteObject, self)._get_format_insert_str(col_names, format_str='?')

class MySQLObject(DBObject):
    def __init__(self):
        super(MySQLObject, self).__init__()

    def _get_create_str(self):
        return 'create database if not exists %s' % self.database_name

    def _cursor_execute(self, cursor):
        try:
            ret = cursor.execute("show databases like '%s'"% self.database_name)
            if not ret:
                cursor.execute(self.create_str)
        except Exception as err:
            pybaselog.error(err)
            return False
        return True

    def _get_insert_str(self, col_names):
        return super(MySQLObject, self)._get_format_insert_str(col_names, format_str='%s')
