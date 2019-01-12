#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyimg_test.py
# Create date : 2018-10-09 17:56
# Modified date : 2018-10-16 12:51
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pybase import pyimg
from pybase import pylog

def _test_sucess():
    url = "http://imgsrc.baidu.com/forum/w%3D580%3B/sign=7dfd5b126a09c93d07f20effaf06f9dc/34fae6cd7b899e51bb592b084fa7d933c9950dd2.jpg"
    (web_status,img_type,img_weight,img_height) = pyimg.get_img_info(url)
    pylog.test_compare_value(web_status, True)
    pylog.test_compare_value(img_type, 'JPEG')
    pylog.test_compare_value(img_weight, 580)
    pylog.test_compare_value(img_height, 580)

def _test_fail():
    url = "http://mhimg1.mnsfz.com/pic/meihuo/2015-2-20/1/13547560956462979517680.jpg"
    (web_status,img_type,img_weight,img_height) = pyimg.get_img_info(url)
    pylog.test_compare_value(web_status, False)
    pylog.test_compare_value(img_type, 0)
    pylog.test_compare_value(img_weight, 0)
    pylog.test_compare_value(img_height, 0)

def test():
    _test_sucess()
    #_test_fail()
