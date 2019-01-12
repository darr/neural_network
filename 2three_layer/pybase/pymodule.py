#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pymodule.py
# Create date : 2016-10-13 14:23
# Modified date : 2018-10-24 21:47
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys
import  os
import platform

if sys.version > '3':
    from pybase import pylog
    from pybase import pypickle
    from pybase import pyfile
    from pybase import etc
else:
    import pylog
    import pypickle
    import pyfile
    import etc

def check_is_install_module(mod_name):
    try:
        __import__(mod_name)
        return True
    except Exception as err:
        pylog.error(err)
        return False

def get_all_modules(name="pip_modules",path=etc.ENV_FILE_PATH):
    con = os.popen("pip list").read()
    path = "%s/python%s/" % (path, platform.python_version())
    pyfile.write_file(con=con, name=name, path=path)

def write_all_modules(name="sys_modules",path=etc.ENV_FILE_PATH):
    mods = sys.modules
    con = ""
    path = "%s/python%s/" % (path, platform.python_version())
    for key in mods:
        con += "%s : %s\n" %(key,mods[key])
    pyfile.write_file(con=con, name=name, path=path)
