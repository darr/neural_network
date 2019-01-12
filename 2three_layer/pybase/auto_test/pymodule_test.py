#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pymodule_test.py
# Create date : 2018-10-09 17:33
# Modified date : 2018-10-24 19:33
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

from pybase import pymodule
from pybase import pylog


def _test_is_install_module():
    modules = ['cStringIO',
                'Image',
                'OpenSSL',
                'Crypto',
                'MySQLdb',
                'sqlite3',
                'zope.interface',
                'pyasn1',
                'twisted',
                'django']

    for each in modules:
        pymodule.check_is_install_module(each)

def _test_check_module():
    module_name = "pylog"
    is_install = pymodule.check_is_install_module(module_name)
    pylog.test_compare_value(is_install,True)

    module_name = "pytest"
    is_install = pymodule.check_is_install_module(module_name)
    pylog.test_compare_value(is_install,True)

def _test_write_all_modules():
    pymodule.write_all_modules()

def _test_get_all_modules():
    pymodule.get_all_modules()

def test():
    #_test_is_install_module()
    _test_check_module()
    _test_write_all_modules()
    _test_get_all_modules()
