#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pydec_encry_test.py
# Create date : 2018-10-19 17:30
# Modified date : 2018-10-19 18:11
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import binascii
from pybase.pyDes import *
from pybase import pydec_encry
from pybase import pylog

def one_test():
    data = "Pleaseencrypt my data"
    k = des("DESCRYPT",CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=PAD_PKCS5)
    d = k.encrypt(data)
    pylog.info("Encrypted:%r" % binascii.hexlify(d))
    pylog.info("Decrypted:%r" % k.decrypt(d))
    pylog.info("data:%s" % data)
    #assert k.decrypt(d) == data

def _test_default():
    content = b"hello world"
    encodeStr = pydec_encry.EncryCode(content)
    real_value = encodeStr
    expect_value = b"QTPkQEyif8xPzalChdHqSQ=="
    pylog.test_compare_value(real_value,expect_value)

    decodeStr = pydec_encry.DecryCode(encodeStr)
    real_value = decodeStr
    expect_value = content
    pylog.test_compare_value(real_value,expect_value)

def _test_self_key():
    content = b"hello world"
    pydec_encry.ENCRY_KEY = "12345678" #need 8 bytes
    encodeStr = pydec_encry.EncryCode(content)
    real_value = encodeStr
    expect_value = b"27Vzf1M8YVkoKD5gjV3wcQ=="
    pylog.test_compare_value(real_value,expect_value)

    decodeStr = pydec_encry.DecryCode(encodeStr)
    real_value = decodeStr
    expect_value = content
    pylog.test_compare_value(real_value,expect_value)

def test():
    #one_test()
    _test_default()
    _test_self_key()
