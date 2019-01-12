#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pysql.py
# Create date : 2016-09-25 09:55
# Modified date : 2018-10-20 13:26
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
import sys

if sys.version > '3':
    from pybase import pylog
else:
    import pylog

DB_TYPE_MYSQL = "mysql"
DB_TYPE_SQLITE = "sqlite"

INT_PRIMARY_KEY = "INT_PRIMARY_KEY"
INT_PRIMARY_KEY_AUTO_INCREMENT = "INT_PRIMARY_KEY_AUTO_INCREMENT"
TYPE_TEXT = "TYPE_TEXT"
TYPE_INT = "TYPE_INT"
TYPE_FLOAT = "TYPE_FLOAT"

def get_mysql_key_type_dic():
    dic = {}
    dic[INT_PRIMARY_KEY_AUTO_INCREMENT] = "int(11) NOT NULL PRIMARY KEY AUTO_INCREMENT"
    dic[INT_PRIMARY_KEY] = "int(11) NOT NULL PRIMARY KEY"
    dic[TYPE_TEXT] = "text "
    dic[TYPE_INT] = "int(11) NOT NULL default 0"
    dic[TYPE_FLOAT] = "FLOAT NOT NULL"
    return dic

def get_sqlite_key_type_dic():
    dic = {}
    dic[INT_PRIMARY_KEY_AUTO_INCREMENT] = "INTEGER PRIMARY KEY   AUTOINCREMENT"
    dic[INT_PRIMARY_KEY] = "INTEGER PRIMARY KEY NOT NULL"
    dic[TYPE_TEXT] = "TEXT"
    dic[TYPE_INT] = "INTEGER"
    dic[TYPE_FLOAT] = "REAL NOT NULL"
    return dic

def get_item_value(key_str, db_type):
    dic = {}
    if db_type == DB_TYPE_MYSQL:
        dic = get_mysql_key_type_dic()
    elif db_type == DB_TYPE_SQLITE:
        dic = get_sqlite_key_type_dic()
    else:
        pylog.error("upkown db_type")

    return dic[key_str]

def get_key_value_str(key, val, db_type):
    key_value_item = ""
    if db_type == DB_TYPE_MYSQL:
        key_value_item = '`%s` %s' %(key, val)
    elif db_type == DB_TYPE_SQLITE:
        key_value_item = '%s %s' %(key, val)
    else:
        pylog.error("upkown db_type")
    return key_value_item

def get_key_value_list_str(table_key_dic, db_type):
    keylist = []
    for key in table_key_dic:
        val = get_item_value(table_key_dic[key], db_type)
        key_value_item = get_key_value_str(key, val, db_type)
        keylist.append(key_value_item)
    key_value_list_str = ',\n'.join(keylist)

    return key_value_list_str

def get_table_create_str(key_value_list_str, tb_name, db_type):
    if db_type == DB_TYPE_MYSQL:
        return "CREATE TABLE IF NOT EXISTS `%s` (\n%s\n ) ENGINE=MyISAM DEFAULT CHARSET=utf8 ;" % (tb_name, key_value_list_str)
    elif db_type == DB_TYPE_SQLITE:
        return "CREATE TABLE IF NOT EXISTS %s(\n%s\n)" % (tb_name, key_value_list_str)
    else:
        pylog.error("upkown db_type")
    return ""

def get_table_create_str_from_dic(table_key_dic, tb_name, db_type):
    key_value_list_str = get_key_value_list_str(table_key_dic, db_type)
    table_create_str = get_table_create_str(key_value_list_str, tb_name, db_type)
    return table_create_str
