#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyfile_test.py
# Create date : 2018-10-09 16:52
# Modified date : 2018-10-15 22:22
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pyfile
from pybase import pylog


def _test_delete_dir():
    path = "./test_pyfile"
    pyfile.delete_dir(path)

def _test_get_file_full_name():
    test_file_path = "./test_pyfile/test"
    test_file_name = "test.py"
    file_name = pyfile.get_file_full_name(test_file_path, test_file_name)
    pylog.test_compare_value(file_name,"./test_pyfile/test/test.py")
    file_object = pyfile.open_file(test_file_path,test_file_name)

def _test_write_read_file():
    write_con = "ok this is my test\n"
    file_object = pyfile.open_file('./test_pyfile/', 'testfile')
    file_object.write(write_con)
    file_object.close()

    file_object = pyfile.open_file('./test_pyfile/', 'testfile',open_type='r')
    con = file_object.read()
    pylog.test_compare_value(con,write_con)

def _test_create_path():
    path = './tmp_path/'
    pyfile.create_path(path)
    pyfile.delete_dir(path)

def test():
    _test_get_file_full_name()
    _test_write_read_file()
    _test_delete_dir()
    _test_create_path()
