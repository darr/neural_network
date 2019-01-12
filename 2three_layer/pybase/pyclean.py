#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyclean.py
# Create date : 2016-08-14 03:00
# Modified date : 2018-10-19 16:38
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import os
import etc

def delete_log_dir():
    os.popen("rm -rf ./log")

def delete_log_db():
    log_db_name = etc.DATABASE_NAME
    os.popen("rm ./db/%s.db" % log_db_name)
    os.popen("rm -rf ./db")

def delete_sqltest_db():
    os.popen("rm ./sqlite_test.db")

def delete_pydb_test_db():
    os.popen("rm ./pydb_test.db")

def delete_pyc():
    os.popen("rm ./*.pyc")

def delete_pycache():
    os.popen("rm -rf ./__pycache__")

def delete_tmp_file():
    tmp_file_name = etc.TMP_FILE_PATH = "./tmp_file/"
    os.popen("rm -rf %s" % tmp_file_name)

if __name__ == "__main__":
    delete_log_dir()
    #delete_log_db()
    #delete_sqltest_db()
    #delete_pydb_test_db()
    delete_pyc()
    delete_tmp_file()
    delete_pycache()
