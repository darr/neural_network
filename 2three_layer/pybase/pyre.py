#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyre.py
# Create date : 2013-12-25 14:47
# Modified date : 2018-10-26 16:50
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import sys
import re

if sys.version > '3':
    from pybase import pylog
else:
    import pylog

def get_center_content(front, behind, content, re_str):
    if not front or not behind or not content or not re_str:
        pylog.condition_value_is_null()
    ret = re.search(r'(?<=%s)%s(?=%s)' % (front, re_str, behind), content, re.S)
    if ret:
        return ret.group()
    return False

def re_find_all_center_content(front, behind, content, re_str):
    ret = re.findall(r'(?<=%s)%s(?=%s)' % (front, re_str, behind), content, re.S)
    if ret:
        return ret
    return False

def delete_html_label(page):
    result, number = re.subn("</p>", "\n", page)
    result, number = re.subn("<br />", "\n", result)
    result, number = re.subn("&nbsp;", " ", result)
    result, number = re.subn("&nbsp;", " ", result)
    result, number = re.subn("&[^;]*;", "", result)
    result, number = re.subn("\n\n", "\n", result)
    result, number = re.subn("<[^>]*>", "", result)
    return result.strip()

def delete_html_tag(page):
    result, number = re.subn("</p>", "\n", page)
    result, number = re.subn("<br />", "\n", result)
    result, number = re.subn("<[^>]*>", "", result)
    return result.strip()

def re_match(match_str, content_str):
    ret = re_search(match_str, content_str)
    if ret:
        return True
    return False

def re_search(match_str, content_str):
    ret = re.search(r'%s' % match_str, content_str)
    return ret

def re_match_first_one_word(content_str):
    p = re.compile("[a-z]+")
    ret = p.match(content_str)
    return ret

def re_match_a_word(word_str, content_str):
    ret = re.match(r'\b%s\b' % word_str, content_str)
    return ret

def re_search_a_str(word_str, content_str):
    ret = re.search(r'\B%s\B' % word_str, content_str)
    return ret

def re_search_a_word(word_str, content_str):
    ret = re.search(r'\b%s\b' % word_str, content_str)
    return ret

def re_search_a_word_space(word_str, content_str):
    ret = re.search(r'\s%s\s' % word_str, content_str)
    return ret

def re_search_a_word_head(word_str, content_str):
    ret = re.search(r'\b%s\B' % word_str, content_str)
    return ret

def re_search_a_word_tail(word_str, content_str):
    ret = re.search(r'\B%s\b' % word_str, content_str)
    return ret
