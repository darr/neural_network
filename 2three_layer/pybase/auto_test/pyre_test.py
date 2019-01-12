#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyre_test.py
# Create date : 2018-10-16 13:39
# Modified date : 2018-10-16 18:36
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import re
from pybase import pyre
from pybase import pylog

def _test_re_match_first_one_word():
    content_str = "string goes here"
    ret = pyre.re_match_first_one_word(content_str)
    real_value = ret.group()
    expect_value = "string"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_match_first_one_word2():
    content_str = "23 45 414 45 "
    ret = pyre.re_match_first_one_word(content_str)
    real_value = ret
    expect_value = None
    pylog.test_compare_value(real_value, expect_value)

def _test_re_search_a_str():
    match_str = "class"
    content_str = "myclassatest"
    ret = pyre.re_search_a_str(match_str, content_str)
    real_value = ret.group()
    expect_value = "class"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_search_re_str():
    match_str = r"b(ab)*"
    content_str = "ababababa"
    ret = pyre.re_search(match_str, content_str)
    real_value = ret.group()
    expect_value = "bababab"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_search_a_word():
    match_str = "class"
    content_str = "my class atest"
    ret = pyre.re_search_a_word(match_str, content_str)
    real_value = ret.group()
    expect_value = "class"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_search_a_word_space():
    match_str = "class"
    content_str = "my class atest"
    ret = pyre.re_search_a_word_space(match_str, content_str)
    real_value = ret.group()
    expect_value = " class "
    pylog.test_compare_value(real_value, expect_value)

def _test_re_search_a_word_head():
    match_str = "class"
    content_str = "my classhead atest"
    ret = pyre.re_search_a_word_head(match_str, content_str)
    real_value = ret.group()
    expect_value = "class"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_search_a_word_tail():
    match_str = "class"
    content_str = "my tailclass atest"
    ret = pyre.re_search_a_word_tail(match_str, content_str)
    real_value = ret.group()
    expect_value = "class"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_match_a_word():
    match_str = "class"
    content_str = "classtest this is "
    ret = pyre.re_match_a_word(match_str, content_str)
    real_value = ret.group()
    expect_value = "class"
    pylog.test_compare_value(real_value, expect_value)

def _test_get_center_content():
    match_str = ".*?"
    front_str = "aaaa"
    behind_str = "ccc"
    content_str = "aaaat1tccc aaaat2tcccxaaaat3tccclll"
    ret = pyre.get_center_content(front_str, behind_str, content_str, match_str)
    real_value = ret
    expect_value = "t1t"
    pylog.test_compare_value(real_value, expect_value)

def _test_get_center_content2():
    match_str = ".*"
    front_str = "aaaa"
    behind_str = "ccc"
    content_str = "aaaat1tccc aaaat2tcccxaaaat3tccclll"
    ret = pyre.get_center_content(front_str, behind_str, content_str, match_str)
    real_value = ret
    expect_value = "t1tccc aaaat2tcccxaaaat3t"
    pylog.test_compare_value(real_value, expect_value)

def _test_re_find_all_center_content():
    match_str = ".*?"
    front_str = "aaaa"
    behind_str = "ccc"
    content_str = "aaaat1tccc aaaat2tcccxaaaat3tccclll"
    ret = pyre.re_find_all_center_content(front_str, behind_str, content_str, match_str)
    real_value = ret
    expect_value = ['t1t', 't2t', 't3t']
    pylog.test_compare_value(real_value, expect_value)

def _test_re_find_all_center_content2():
    match_str = ".*"
    front_str = "aaaa"
    behind_str = "ccc"
    content_str = "aaaat1tccc aaaat2tcccxaaaat3tccclll"
    ret = pyre.re_find_all_center_content(front_str, behind_str, content_str, match_str)
    real_value = ret
    expect_value = ['t1tccc aaaat2tcccxaaaat3t']
    pylog.test_compare_value(real_value, expect_value)

def test():
    _test_re_match_first_one_word()
    _test_re_match_first_one_word2()
    _test_re_search_a_str()
    _test_re_search_re_str()
    _test_re_search_a_word()
    _test_re_search_a_word_space()
    _test_re_search_a_word_head()
    _test_re_search_a_word_tail()
    _test_get_center_content()
    _test_get_center_content2()
    _test_re_find_all_center_content()
    _test_re_find_all_center_content2()
