#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pypickle.py
# Create date : 2014-01-12 10:52
# Modified date : 2018-10-24 19:43
# Author : DARREN
# Describe :序列化数据到本地
# Email : lzygzh@126.com
#####################################

import sys
import pickle
import hashlib

if sys.version > '3':
    from pybase import pyfile
    from pybase import pylog
    from pybase import etc
else:
    import pyfile
    import pylog
    import etc

def _get_save_path():
    return etc.TMP_FILE_PATH

def _get_md5_name(name):
    md5_value = hashlib.md5(name.encode("utf-8"))
    md5_value.digest()
    md5_str = "%s" % md5_value.hexdigest()
    return md5_str

def _get_write_save_file(path, name):
    obj = pyfile.open_file(path, name, 'wb')
    return obj

def _get_read_save_file(path, name):
    obj = pyfile.open_file(path, name, 'rb')
    return obj

def save(content, name, path=_get_save_path()):
    try:
        md5_str = _get_md5_name(name)
        file_obj = _get_write_save_file(path, md5_str)
        pickle.dump(content, file_obj, 0)
    except Exception as err:
        pylog.error(err)
        return False
    return True

def read(name, path=_get_save_path()):
    try:
        md5_str = _get_md5_name(name)
        file_obj = _get_read_save_file(path, md5_str)
        obj = pickle.load(file_obj)
        return obj
    except Exception as err:
        pylog.error(err)
        return False
