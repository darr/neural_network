#!/usr/bin/python
# -*- coding: utf-8 -*-
########################################
# File name : pyjson.py
# Create date : 2013-06-22 11:06
# Modified date : 2018-12-24 20:04
# Author : DARREN
# Email : lzygzh@126.com
########################################
from __future__ import division
from __future__ import print_function

import sys
import json

if sys.version > '3':
    from pybase import pyjson_data
else:
    import pyjson_data

def get_request_get():
    return pyjson_data.REQUEST_GET

def get_request_post():
    return pyjson_data.REQUEST_POST

def get_method_not_support_content():
    return pyjson_data.REQUEST_METHOD_NOT_SUPPORT

def get_str_no_description():
    return pyjson_data.NO_DESCRIPTION

def json_ok(data_content, request_method=get_request_get()):
    ret_type = pyjson_data.JSON_TYPE_OK
    return json_ret(ret_type, data_content, request_method)

def json_fail(data_content, request_method=get_request_get()):
    ret_type = pyjson_data.JSON_TYPE_FAIL
    return json_ret(ret_type, data_content, request_method)

def json_method_not_support(data_content=get_method_not_support_content(), request_method=get_request_get()):
    ret_type = pyjson_data.JSON_TYPE_FAIL
    return json_ret(ret_type, data_content, request_method)

def json_to_dict(json_str):
    if sys.version > '3':
        #dic = json.loads(json_str.decode('utf-8'))
        dic = json.loads(json_str)
    else:
        dic = json.loads(json_str)
    return dic

def dict_to_json(dic):
    json_str = json.dumps(dic)
    return json_str

def json_ret(ret_type, data_content,request_method):
    Dic = { pyjson_data.JSON_KEY_RESULT: ret_type,
            pyjson_data.JSON_KEY_DATA: data_content,
            pyjson_data.REQUEST_METHOD: request_method
            }
    jsn = json.dumps(Dic)
    return jsn
