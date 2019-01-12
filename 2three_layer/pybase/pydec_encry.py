#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pydec_encry.py
# Create date : 2014-10-18 18:37
# Modified date : 2018-10-20 12:50
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import base64
from pyDes import *
import binascii

#ENCRY_KEY = "DESCRYPT"  #这个必须要8个bytes
#ENCRY_KEY = "12345678"  #这个必须要8个bytes
ENCRY_KEY = "distance"  #这个必须要8个bytes


def _EncryCodeWithKey(key,con_str):
    con_str = _desEncryWithKey(key,con_str)
    con_str = base64.b64encode(con_str)
    return con_str

def _desEncryWithKey(key,con_str):
    #DesObject = des(key, CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=PAD_PKCS5)
    DesObject = des(key, CBC, "\1\2\3\4\5\6\7\7", pad=None, padmode=PAD_PKCS5)
    desEncStr = DesObject.encrypt(con_str)
    return desEncStr

def _DecryCodeWithKey(key,con_str):
    con_str = base64.b64decode(con_str)
    return _desDecryWithKey(key,con_str)

def _desDecryWithKey(key,con_str):
    #DesObject = des(key, CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=PAD_PKCS5)
    DesObject = des(key, CBC, "\1\2\3\4\5\6\7\7", pad=None, padmode=PAD_PKCS5)
    desDecStr = DesObject.decrypt(con_str)
    return desDecStr

def EncryCode(con_str):
    return _EncryCodeWithKey(ENCRY_KEY,con_str)

def DecryCode(con_str):
    return _DecryCodeWithKey(ENCRY_KEY,con_str)

