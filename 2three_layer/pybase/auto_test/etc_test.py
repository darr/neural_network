#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc_test.py
# Create date : 2018-10-07 11:28
# Modified date : 2018-10-20 14:46
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
import etc
import pylog


def test():
    pylog.test_compare_value(etc.LOG_PRINT_LEVEL, "info")
    pylog.test_compare_value(etc.LOG_PRINT_ALL_INFO_LEVEL, "warning")

    pylog.test_compare_value(etc.LOG_FILE_NAME, "logfile")
    pylog.test_compare_value(etc.LOG_FILE_PATH, "./log/")

    pylog.test_compare_value(etc.LOG_WRITE_LEVEL, "debug")
    pylog.test_compare_value(etc.LOG_WRITE_DB_LEVEL, "warning")
    pylog.test_compare_value(etc.LOG_LEVEL_DIC, {})
    pylog.test_compare_value(etc.IS_INIT, False)
    pylog.test_compare_value(etc.TMP_FILE_PATH, "./tmp_file/")

    pylog.test_compare_value(etc.PYLOG_DATABASE_NAME, "pylog")
    pylog.test_compare_value(etc.PYLOG_DATABASE_PATH, "./db/")
    pylog.test_compare_value(etc.PYLOG_DATABASE_HOST, "localhost")
    pylog.test_compare_value(etc.PYLOG_TABLE_NAME, "pylog")
    pylog.test_compare_value(etc.PYLOG_DATABASE_CHARSET, "utf8")
    pylog.test_compare_value(etc.DATABASE_USE, "you")
    pylog.test_compare_value(etc.DATABASE_HOST, "localhost")
    pylog.test_compare_value(etc.DATABASE_PASSWORD, "888888")
    pylog.test_compare_value(etc.DATABASE_CHARSET, "utf8")
    pylog.test_compare_value(etc.DATABASE_NAME, "dbname")
    pylog.test_compare_value(etc.DATABASE_TABLE_NAME, "table_test")
