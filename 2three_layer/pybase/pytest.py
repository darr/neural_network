#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pytest.py
# Create date : 2018-10-07 10:58
# Modified date : 2018-10-27 13:26
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("..")
import pylinux

from auto_test import etc_test
from auto_test import pylog_test
from auto_test import pybaselog_test
from auto_test import pyfile_test
from auto_test import pymodule_test
from auto_test import pytime_test
from auto_test import pyimg_test
from auto_test import pyjson_data_test
from auto_test import pyjson_test
from auto_test import pylinux_test
from auto_test import pyobject_test
from auto_test import pybiology_test
from auto_test import pyabc_test
from auto_test import pypickle_test
from auto_test import pyhttp_test
from auto_test import pyprocess_test
from auto_test import pyre_test
from auto_test import pysqlite_test
from auto_test import pysql_test
from auto_test import pymysqldb_test
from auto_test import pydb_test
from auto_test import pylog_table_test
from auto_test import pydec_encry_test
from auto_test import pyserver_test
from auto_test import pymodule_test
from auto_test import pysig_test
from auto_test import pythread_test

def show_env():
    print(pylinux.get_system_name_version())
    print(pylinux.get_platform_unname()[3])
    print(pylinux.get_architecture())
    print("Python:%s" % sys.version)

def getAllModule():
    module_str = os.popen("pip list").read()
    lt = module_str.split('\n')
    for i in lt:
        print(i)

def run():
    show_env()
    #getAllModule()
    #pylog_test.test()
    #pybaselog_test.test()
    etc_test.test()
    pyfile_test.test()
    pymodule_test.test()
    pytime_test.test()
    pyimg_test.test()
    pyjson_data_test.test()
    pyjson_test.test()
    pylinux_test.test()
    pyobject_test.test()
    pybiology_test.test()
    pyabc_test.test()
    pypickle_test.test()
    pyhttp_test.test()
    pyprocess_test.test()
    pyre_test.test()
    pysqlite_test.test()
    pysql_test.test()
    pymysqldb_test.test()
    pydb_test.test()
    pylog_table_test.test()
    pydec_encry_test.test()
    pyserver_test.test()
    pysig_test.test()
    pythread_test.test()


if __name__ == "__main__":
    run()
