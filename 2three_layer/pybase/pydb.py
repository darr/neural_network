#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pydb.py
# Create date : 2016-09-25 05:34
# Modified date : 2018-10-20 14:53
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import sys

if sys.version > '3':
    from pybase import pyobject
    from pybase import pymysqldb
    from pybase import pysqlite
    from pybase import pybaselog
    from pybase import pyfile
    from pybase import etc
    from pybase import pysql
else:
    import pyobject
    import pymysqldb
    import pysqlite
    import pybaselog
    import pyfile
    import etc
    import pysql

def get_charset():
    return etc.DATABASE_CHARSET

# pylint: disable=bad-continuation
# pylint: disable=attribute-defined-outside-init
class PyTable(pyobject.BaseObject):
    def __init__(self):
        super(PyTable, self).__init__()
        self.tb_charset = get_charset()

    def _get_mysql_table(self):
        mysql_tb = pymysqldb.SQLTable(
                                    database_name=self.db_name,
                                    table_name=self.tb_name,
                                    host=self.db_host,
                                    user=self.user_name,
                                    password=self.user_password,
                                    charset=self.tb_charset,
                                    create_table_str=self.create_str,
                                    )
        return mysql_tb

    def _get_sqlite_table(self):
        pyfile.create_path(self.db_path)
        self.db_full_name = "%s%s" % (self.db_path, self.db_name)
        sqlite_tb = pysqlite.SQLiteTable(
                                        table_name=self.tb_name,
                                        database_name=self.db_full_name,
                                        create_table_str=self.create_str,
                                        charset=self.tb_charset,
                                        )
        return sqlite_tb

    def create_mysql_table_with_dic(self,
                                    table_dic,
                                    database_name,
                                    table_name,
                                    database_host,
                                    user,
                                    database_password):
        table_create_str = pysql.get_table_create_str_from_dic(table_dic,
                                                               table_name,
                                                               pysql.DB_TYPE_MYSQL)
        self.create_mysql_table(database_name,
                                table_name,
                                database_host,
                                user,
                                database_password,
                                table_create_str)

    def create_mysql_table(self,
                           database_name,
                           table_name,
                           database_host,
                           user,
                           database_password,
                           table_create_str):
        self.db_type = "mysql"
        self.db_name = database_name
        self.tb_name = table_name
        self.db_host = database_host
        self.user_name = user
        self.user_password = database_password
        self.create_str = table_create_str

        self.table = self._get_mysql_table()


    def create_sqlite_table_with_dic(self,
                                     table_dic,
                                     database_name,
                                     table_name,
                                     database_path):
        table_create_str = pysql.get_table_create_str_from_dic(table_dic,
                                                              table_name,
                                                              pysql.DB_TYPE_SQLITE)
        self.create_sqlite_table(database_name,
                                 table_name,
                                 table_create_str,
                                 database_path)

    def create_sqlite_table(self,
                            database_name,
                            table_name,
                            table_create_str,
                            database_path):
        self.db_type = "sqlite"
        self.db_name = database_name
        self.tb_name = table_name
        self.db_path = database_path
        self.create_str = table_create_str
        self.table = self._get_sqlite_table()


    def insert(self, item_dic):
        if not item_dic:
            pybaselog.condition_value_is_null()

        last_id = self.table.insert_item_with_dic(item_dic)
        return last_id

    def update(self, update_str, key_str):
        if not update_str or not key_str:
            pybaselog.condition_value_is_null()
        return self.table.update_item(update_str=update_str, key_str=key_str)

    def query_item(self,
                   search_content='*',
                   search_condition='',
                   page='1',
                   page_size='1',
                   order_by="order by ut desc"):
        query_content = self.table.query_item(search_content=search_content,
                                              search_condition=search_condition,
                                              page=page,
                                              page_size=page_size,
                                              order_by=order_by)
        return query_content

    def query_count(self, search_content="count(*)", search_condition=""):
        return self.table.query_count(search_content, search_condition)

    def delete_item(self, delete_condition_str):
        return self.table.delete_item(delete_condition_str)

    def drop_table(self):
        return self.table._drop_table()

    def drop_database(self):
        return self.table._drop_database()
# pylint: enable=attribute-defined-outside-init
# pylint: enable=bad-continuation
